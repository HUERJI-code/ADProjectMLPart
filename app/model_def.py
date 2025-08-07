import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_user_tags, num_items, content_dim, embedding_dim=128, user_feature_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.user_feature_dim = user_feature_dim

        # 用户特征分支
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_tag_embedding = nn.Embedding(num_user_tags, embedding_dim)  # 稠密嵌入
        self.user_tag_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embedding_dim)
        )
        self.user_feature_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, user_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(user_feature_dim)
        )

        # 物品特征分支
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.content_fc = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(embedding_dim)
        )
        self.item_feature_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(embedding_dim)
        )

        # 协同过滤网络
        self.cf_layers = nn.Sequential(
            nn.Linear(embedding_dim + user_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.predict_layer = nn.Linear(64, 1)

        # 相似用户表示
        self.user_similarity_layer = nn.Sequential(
            nn.Linear(user_feature_dim, user_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(user_feature_dim)
        )

    def forward(self, user_ids, user_tags_idx, item_ids, content_features):
        user_id_emb = self.user_embedding(user_ids)

        # 标签稠密嵌入（多个标签求平均）
        tag_emb = self.user_tag_embedding(user_tags_idx)  # (B, T, D)
        tag_emb = tag_emb.mean(dim=1)  # 均值池化
        tag_emb = self.user_tag_proj(tag_emb)  # 稠密映射

        user_combined = torch.cat([user_id_emb, tag_emb], dim=-1)
        user_emb_enhanced = self.user_feature_layer(user_combined)

        item_id_emb = self.item_embedding(item_ids)
        content_emb = self.content_fc(content_features)
        item_emb = torch.cat([item_id_emb, content_emb], dim=-1)
        item_emb = self.item_feature_layer(item_emb)

        x = torch.cat([user_emb_enhanced, item_emb], dim=-1)
        cf_out = self.cf_layers(x)
        pred = torch.sigmoid(self.predict_layer(cf_out))
        return pred.squeeze()

    def get_user_representation(self, user_ids, user_tags_idx):
        user_id_emb = self.user_embedding(user_ids)
        tag_emb = self.user_tag_embedding(user_tags_idx).mean(dim=1)
        tag_emb = self.user_tag_proj(tag_emb)
        user_combined = torch.cat([user_id_emb, tag_emb], dim=-1)
        user_emb = self.user_feature_layer(user_combined)
        return self.user_similarity_layer(user_emb)