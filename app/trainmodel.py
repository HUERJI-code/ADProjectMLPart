# trainmodel.py

import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from app.model_def import HybridRecommender

# —— 1. 建立数据库连接 —— 
DB_URI = "mysql+pymysql://root@localhost/adproject"
engine = create_engine(DB_URI)

def load_data():
    # —— 1.1 读取 activities 表 —— 
    activity_df = pd.read_sql("""
        SELECT
          ActivityId  AS activityid,
          Title       AS title,
          Description AS description,
          Location    AS content,
          StartTime   AS starttime,
          EndTime     AS endtime,
          number      AS duration
        FROM activities
    """, engine)

    # —— 1.2 解析时间字段 —— 
    activity_df['starttime'] = pd.to_datetime(
        activity_df['starttime'], errors='coerce'
    )
    activity_df['endtime'] = pd.to_datetime(
        activity_df['endtime'], errors='coerce'
    )

    # —— 1.3 读取 userfavouriteactivity 表 ——
    interactions_df = pd.read_sql("""
        SELECT
          UserId     AS userid,
          ActivityId AS activityid,
          1          AS favorite
        FROM userfavouriteactivity
    """, engine)

    # —— 1.4 读取 users 表 ——
    user_df = pd.read_sql("""
        SELECT
          UserId AS userid,
          Name,
          Email
        FROM users
    """, engine)

    # —— 1.5 读取 profile-标签关联，并聚合到每个用户 ——
    userprofiles_df = pd.read_sql(
        "SELECT Id AS profile_id, UserId FROM userprofiles", engine
    )
    upt = pd.read_sql(
        "SELECT UserProfileId AS profile_id, TagId FROM userprofiletag", engine
    )
    tags_df = pd.read_sql(
        "SELECT TagId, Name FROM tags", engine
    )

    # 合并拿到每条 (UserId, tagName)
    upt = upt.merge(tags_df, on='TagId', how='left')
    upt = upt.merge(userprofiles_df, on='profile_id', how='right')

    # 聚合成 "tag1,tag2" 字符串
    tag_map = (
        upt.groupby('UserId')['Name']
        .apply(lambda names: ','.join(names.dropna().unique().astype(str)))
        .to_dict()
    )
    user_df['tags'] = user_df['userid'].map(tag_map).fillna('')

    # —— 1.6 构造交互矩阵 ——
    interaction_matrix = interactions_df.pivot_table(
        index='userid',
        columns='activityid',
        values='favorite',
        aggfunc='max'
    ).fillna(0).astype(int)

    # —— 1.7 补全所有出现过的用户，即使他们没 profile/标签 ——
    missing = set(interaction_matrix.index) - set(user_df['userid'])
    if missing:
        extra = pd.DataFrame({
            'userid': list(missing),
            'Name': [''] * len(missing),
            'Email': [''] * len(missing),
            'tags': [''] * len(missing)
        })
        user_df = pd.concat([user_df, extra], ignore_index=True)

    # 按交互矩阵用户顺序重排 user_df
    user_df = user_df.set_index('userid') \
        .reindex(interaction_matrix.index) \
        .reset_index()

    # —— 1.8 用户历史记录字典 ——
    user_history = {
        uid: interaction_matrix.loc[uid].to_dict()
        for uid in interaction_matrix.index
    }

    return activity_df, interactions_df, user_df, interaction_matrix, user_history


# ===== 2. 特征工程 =====

class ActivityFeatureEngineer:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=300, stop_words='english'
        )
        self.tag_encoder     = MultiLabelBinarizer()

    def preprocess_tags(self, tags):
        if pd.isna(tags) or not tags:
            return []
        return [
            t.strip()
            for t in re.split(r'[/,]', tags)
            if t.strip()
        ]

    def extract_features(self, df):
        df['combined_text'] = (
                df['title'].fillna('') + ' '
                + df['description'].fillna('') + ' '
                + df['content'].fillna('')
        )
        tfidf    = self.text_vectorizer.fit_transform(df['combined_text'])
        # 暂不使用活动标签，保持占位
        df['tag_list'] = df['combined_text'].apply(lambda _: [])
        tags_mat = self.tag_encoder.fit_transform(df['tag_list'])

        df['hour']     = df['starttime'].dt.hour.fillna(0).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        time_feat     = df[['hour_sin', 'hour_cos', 'duration']].values

        features = np.hstack([tfidf.toarray(), tags_mat, time_feat])
        return features, self.tag_encoder.classes_


class UserFeatureEngineer:
    def __init__(self):
        self.tag_encoder = MultiLabelBinarizer()

    def preprocess_tags(self, tags):
        if pd.isna(tags) or not tags:
            return []
        return [
            t.strip()
            for t in re.split(r'[/,]', tags)
            if t.strip()
        ]

    def extract_user_features(self, user_df):
        user_df['tag_list'] = user_df['tags'].apply(self.preprocess_tags)
        one_hot            = self.tag_encoder.fit_transform(user_df['tag_list'])
        tag_to_idx         = {
            t: i
            for i, t in enumerate(self.tag_encoder.classes_)
        }
        user_tag_indices = [
            [tag_to_idx[t] for t in tags if t in tag_to_idx]
            for tags in user_df['tag_list']
        ]
        return user_tag_indices, self.tag_encoder.classes_, tag_to_idx


# ===== 3. Dataset & 4. 训练 =====

class CustomDataset(Dataset):
    def __init__(self, interaction_matrix, activity_features, user_tag_indices):
        self.users, self.items, self.labels = [], [], []
        self.activity_features = activity_features
        self.user_tag_indices  = user_tag_indices
        self.user_map = {
            u: i
            for i, u in enumerate(interaction_matrix.index)
        }
        self.item_map = {
            i: j
            for j, i in enumerate(interaction_matrix.columns)
        }

        for uid in interaction_matrix.index:
            uidx = self.user_map[uid]
            for aid in interaction_matrix.columns:
                iidx = self.item_map[aid]
                val  = interaction_matrix.at[uid, aid]
                lbl  = 1 if val in (1, 2) else 0
                self.users.append(uidx)
                self.items.append(iidx)
                self.labels.append(lbl)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uidx      = self.users[idx]
        iidx      = self.items[idx]
        tags      = self.user_tag_indices[uidx][:10]
        tag_tensor= torch.zeros(10, dtype=torch.long)
        tag_tensor[:len(tags)] = torch.tensor(tags, dtype=torch.long)
        return (
            torch.tensor(uidx, dtype=torch.long),
            tag_tensor,
            torch.tensor(iidx, dtype=torch.long),
            torch.tensor(self.activity_features[iidx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for uids, tags, iids, feats, lbls in train_loader:
            uids, tags, iids, feats, lbls = [
                x.to(device)
                for x in (uids, tags, iids, feats, lbls)
            ]
            optimizer.zero_grad()
            preds = model(uids, tags, iids, feats)
            loss  = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss/len(train_loader):.4f}")
    return model


def should_update_model(old_uids, old_iids, new_uids, new_iids):
    return True


# ===== 5. 主流程 =====

if __name__ == "__main__":
    act_df, inter_df, user_df, inter_mat, user_hist = load_data()

    act_feats, act_tags = ActivityFeatureEngineer().extract_features(act_df)
    user_idxs, user_tags, tag_map = UserFeatureEngineer().extract_user_features(user_df)

    dataset     = CustomDataset(inter_mat, act_feats, user_idxs)
    train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=128)

    # —— 保底 num_user_tags ≥ 1 ——
    safe_tags = max(1, len(user_tags))

    model = HybridRecommender(
        num_users     = len(inter_mat.index),
        num_user_tags = safe_tags,
        num_items     = len(act_df),
        content_dim   = act_feats.shape[1]
    )

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    ckpt_path = model_dir / "recommender_checkpoint.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not should_update_model(
                ckpt['user_idx_map'].keys(),
                ckpt['item_idx_map'].keys(),
                inter_mat.index,
                inter_mat.columns
        ):
            print("✅ 无需更新模型")
            exit()

    trained = train_model(model, train_loader, val_loader, epochs=100)

    # —— 保存 checkpoint，新增几个字段 ——
    torch.save({
        "model_state_dict": trained.state_dict(),
        "model_config": {
            "num_users": len(inter_mat.index),
            "num_user_tags": safe_tags,
            "num_items": len(act_df),
            "content_dim": act_feats.shape[1]
        },
        "user_tag_encoder": tag_map,
        "user_tag_indices": user_idxs,
        "events_info": act_df,
        "user_history_dict": user_hist,
        "activity_features": act_feats,
        "user_idx_map": {u: i for i, u in enumerate(user_df['userid'])},
        "item_idx_map": {i: j for j, i in enumerate(act_df['activityid'])},
    }, ckpt_path)

    print("✅ 模型训练完成并保存！")
