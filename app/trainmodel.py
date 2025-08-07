import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
import re
from datetime import datetime
from app.model_def import HybridRecommender

# ===== 1. 数据加载与预处理 =====
def load_data(activity_path, interaction_path, user_path):
    activity_df = pd.read_csv(activity_path)
    interactions_df = pd.read_csv(interaction_path)
    user_df = pd.read_csv(user_path)

    activity_df['starttime'] = pd.to_datetime(activity_df['starttime'])
    activity_df['endtime'] = pd.to_datetime(activity_df['endtime'])
    activity_df['duration'] = (activity_df['endtime'] - activity_df['starttime']).dt.total_seconds() / 3600

    # 构建交互矩阵（DataFrame）
    interaction_matrix = interactions_df.pivot_table(
        index='userid',
        columns='activityid',
        values='favorite',
        aggfunc='max'
    ).fillna(0).astype(int)

    # 构建用户历史字典（Dict）
    user_history_dict = interactions_df[interactions_df['favorite'] > 0].groupby('userid')['activityid'].apply(list).to_dict()

    return activity_df, interactions_df, user_df, interaction_matrix, user_history_dict

# ===== 2. 特征工程类 =====
class ActivityFeatureEngineer:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        self.tag_encoder = MultiLabelBinarizer()

    def preprocess_tags(self, tags):
        if pd.isna(tags): return []
        return [tag.strip() for tag in re.split(r'[/,]', tags) if tag.strip()]

    def extract_features(self, df):
        df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['content'].fillna('')
        tfidf = self.text_vectorizer.fit_transform(df['combined_text'])

        df['tag_list'] = df['tag'].apply(self.preprocess_tags)
        tags = self.tag_encoder.fit_transform(df['tag_list'])

        df['hour'] = df['starttime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        time_feat = df[['hour_sin', 'hour_cos', 'duration']].values

        features = np.hstack([tfidf.toarray(), tags, time_feat])
        return features, self.tag_encoder.classes_

class UserFeatureEngineer:
    def __init__(self):
        self.tag_encoder = MultiLabelBinarizer()

    def preprocess_tags(self, tags):
        if pd.isna(tags): return []
        return [tag.strip() for tag in re.split(r'[/,]', tags) if tag.strip()]

    def extract_user_features(self, user_df):
        user_df['tag_list'] = user_df['tags'].apply(self.preprocess_tags)
        one_hot_tags = self.tag_encoder.fit_transform(user_df['tag_list'])
        tag_to_idx = {tag: i for i, tag in enumerate(self.tag_encoder.classes_)}
        user_tag_indices = [
            [tag_to_idx[tag] for tag in tags if tag in tag_to_idx]
            for tags in user_df['tag_list']
        ]
        return user_tag_indices, self.tag_encoder.classes_, tag_to_idx

# ===== 3. Dataset 类 =====
class CustomDataset(Dataset):
    def __init__(self, interaction_matrix, activity_features, user_tag_indices):
        self.users, self.items, self.labels = [], [], []
        self.activity_features = activity_features
        self.user_tag_indices = user_tag_indices
        self.user_map = {u: i for i, u in enumerate(interaction_matrix.index)}
        self.item_map = {i: j for j, i in enumerate(interaction_matrix.columns)}

        for user_id in interaction_matrix.index:
            user_idx = self.user_map[user_id]
            for item_id in interaction_matrix.columns:
                item_idx = self.item_map[item_id]
                interaction_value = interaction_matrix.at[user_id, item_id]

                if interaction_value in [1, 2]:
                    label = 1
                else:
                    label = 0

                self.users.append(user_idx)
                self.items.append(item_idx)
                self.labels.append(label)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_idx = self.users[idx]
        user_tags = self.user_tag_indices[user_idx]
        max_len = 10
        tag_tensor = torch.zeros(max_len, dtype=torch.long)
        tag_tensor[:len(user_tags[:max_len])] = torch.tensor(user_tags[:max_len])

        return (
            torch.tensor(user_idx, dtype=torch.long),
            tag_tensor,
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.activity_features[self.items[idx]], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

# ===== 4. 训练主函数 =====
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 关键修正：匹配数据集的返回项
        for user_ids, user_tags, item_ids, content, label in train_loader:
            user_ids, user_tags, item_ids, content, label = (
                user_ids.to(device),
                user_tags.to(device),
                item_ids.to(device),
                content.to(device),
                label.to(device)
            )

            optimizer.zero_grad()
            output = model(user_ids, user_tags, item_ids, content)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model


# ===== 5. 检查是否需更新模型 =====
def should_update_model(old_user_ids, old_item_ids, new_user_ids, new_item_ids):
    return (True)#len(set(new_user_ids) - set(old_user_ids)) > 0 or len(set(new_item_ids) - set(old_item_ids)) >= 10)

# ===== 6. 主流程 =====
if __name__ == "__main__":
    act_file = "../data/activity.csv"
    int_file = "../data/useractivityevaluations.csv"
    user_file = "../data/user.csv"
    model_path = "../model/recommender_checkpoint.pth"

    # 加载活动数据和用户数据
    events_df, interactions_df, user_df, interaction_matrix, user_history = load_data(act_file, int_file, user_file)

    # 特征工程
    activity_engineer = ActivityFeatureEngineer()
    user_engineer = UserFeatureEngineer()

    activity_features, activity_tag_classes = activity_engineer.extract_features(events_df)
    user_tag_indices, user_tag_classes, tag_to_idx = user_engineer.extract_user_features(user_df)

    # 创建数据集
    dataset = CustomDataset(interaction_matrix, activity_features, user_tag_indices)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    # 模型初始化
    model = HybridRecommender(
        num_users=len(interaction_matrix.index),
        num_user_tags=len(user_tag_classes),
        num_items=len(events_df),  # 保证用全量活动数量
        content_dim=activity_features.shape[1]
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if not should_update_model(
                checkpoint['user_idx_map'].keys(),
                checkpoint['item_idx_map'].keys(),
                interaction_matrix.index,
                interaction_matrix.columns
        ):
            print("✅ 无需更新模型")
            exit()

    trained_model = train_model(model, train_loader, val_loader, epochs=100)
    os.makedirs("../model", exist_ok=True)

    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "model_config": {
            "num_users": len(interaction_matrix.index),
            "num_user_tags": len(user_tag_classes),
            "num_items": len(events_df),
            "content_dim": activity_features.shape[1],
            "embedding_dim": 128,
            "user_feature_dim": 64
        },
        "user_tag_encoder": user_engineer.tag_encoder,
        "tag_to_idx": tag_to_idx,
        "activity_features": activity_features,
        "user_idx_map": {u: i for i, u in enumerate(user_df['userid'])},
        "item_idx_map": {i: j for j, i in enumerate(events_df['id'])},
        "events_info": events_df,
        "user_history_dict": user_history,
        "user_tag_indices": user_tag_indices
    }, model_path)



    print("✅ 模型训练完成并保存！")