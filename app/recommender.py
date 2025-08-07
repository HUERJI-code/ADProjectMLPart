import numpy as np
import torch
import pandas as pd
import os
from app.model_def import HybridRecommender

@torch.no_grad()
def load_model(checkpoint_path="D:/study/nus/CapstoneProject/ActivityRecommend/app/model/recommender_checkpoint.pth"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("模型未训练，请先运行 train_model.py")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint['model_config']
    model = HybridRecommender(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return (
        model,
        checkpoint['activity_features'],
        checkpoint['user_tag_indices'],
        checkpoint['user_idx_map'],
        checkpoint['item_idx_map'],
        checkpoint['events_info'],
        checkpoint['user_history_dict'],
        checkpoint['user_tag_encoder']
    )

@torch.no_grad()
def recommend_activities(user_id, top_k):
    model, activity_features, user_tag_indices, user_idx_map, item_idx_map, events_df, history_dict, _ = load_model()

    if user_id not in user_idx_map:
        raise ValueError(f"用户 {user_id} 不存在于训练数据中")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    user_idx = user_idx_map[user_id]
    user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)

    # 标签 index 处理
    tag_ids = user_tag_indices[user_idx]
    tag_len = len(tag_ids)
    max_len = max(tag_len, 1)
    tag_tensor = torch.zeros(1, max_len, dtype=torch.long).to(device)
    tag_tensor[0, :tag_len] = torch.tensor(tag_ids[:max_len]).to(device)

    n_items = len(events_df)
    all_item_idx = torch.arange(n_items, dtype=torch.long).to(device)
    content_tensor = torch.tensor(activity_features, dtype=torch.float).to(device)

    repeated_user = user_idx_tensor.repeat(n_items)
    repeated_tag = tag_tensor.repeat(n_items, 1)

    scores = model(repeated_user, repeated_tag, all_item_idx, content_tensor)

    fav_dict = history_dict.get(user_id, {})

    # —— 关键修改：使用实际活动数生成掩码 ——
    valid_mask = [i not in fav_dict for i in range(n_items)]
    valid_mask = np.array(valid_mask)

    scores = scores[valid_mask]
    top_scores, top_idx = torch.topk(scores, k=min(top_k, scores.shape[0]))

    valid_indices = all_item_idx[valid_mask].cpu().numpy()
    top_idx = top_idx.cpu().numpy()

    recommended = events_df.iloc[valid_indices[top_idx]].copy()
    recommended['score'] = top_scores.cpu().numpy()

    recommended = recommended.rename(columns={'activityid': 'id'})
    return recommended[['id', 'title', 'score']]


@torch.no_grad()
def recommend_similar_users(user_id, top_k=5):
    model, _, user_tag_indices, user_idx_map, _, _, _, _ = load_model()

    if user_id not in user_idx_map:
        raise ValueError(f"用户 {user_id} 不存在于训练数据中")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    user_count = len(user_idx_map)
    all_user_idx = torch.arange(user_count, dtype=torch.long).to(device)

    # 构造用户标签矩阵 (B, T)
    max_len = 60
    all_user_tags = torch.zeros(user_count, max_len, dtype=torch.long)
    for idx in range(user_count):
        tag_ids = user_tag_indices[idx][:max_len]
        all_user_tags[idx, :len(tag_ids)] = torch.tensor(tag_ids)

    all_user_tags = all_user_tags.to(device)

    user_idx = torch.tensor([user_idx_map[user_id]], dtype=torch.long).to(device)
    user_tag = all_user_tags[user_idx].to(device)

    all_reps = model.get_user_representation(all_user_idx, all_user_tags)
    target = model.get_user_representation(user_idx, user_tag)

    sims = torch.nn.functional.cosine_similarity(target, all_reps, dim=1)
    sims[user_idx_map[user_id]] = -1  # 排除自己

    top_sim, top_idx = torch.topk(sims, k=top_k)
    idx2uid = {v: k for k, v in user_idx_map.items()}
    return [(idx2uid[i.item()], top_sim[j].item()) for j, i in enumerate(top_idx)]

if __name__ == "__main__":
    uid = 11
    print("\n🔥 推荐活动:")
    print(recommend_activities(uid,5))

    print("\n👥 相似用户:")
    for i, (u, s) in enumerate(recommend_similar_users(uid), 1):
        print(f"{i}. 用户 {u}, 相似度 {s:.4f}")
