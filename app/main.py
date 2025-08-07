# app/main.py

# python -m uvicorn app.main:app --reload --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
from app.recommender import recommend_activities, recommend_similar_users

app = FastAPI(
    title="活动推荐系统",
    description="基于混合模型的推荐接口",
    version="1.0"
)


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10


class SimilarUserRequest(BaseModel):
    user_id: int
    top_k: int = 5


@app.post("/recommend/")
def recommend(request: RecommendRequest):
    # 调用推荐函数，返回一个包含 id, title, score 的 DataFrame
    df = recommend_activities(request.user_id, request.top_k)
    # 只取 id 列，并转成 int 列表
    activity_ids = df['id'].astype(int).tolist()
    return {
        # "user_id": request.user_id,
        "recommended_activity_ids": activity_ids
    }



@app.post("/similar-users/")
def similar_users(request: SimilarUserRequest):
    raw = recommend_similar_users(request.user_id, request.top_k)
    # 过滤并转换成纯 Python int/float
    sim_users = []
    for uid, score in raw:
        try:
            uid_int = int(uid)
            score_f = float(score)
        except (ValueError, TypeError):
            # 跳过无法转换的条目
            continue
        sim_users.append({
            "user_id": uid_int
        })
    return {
        "user_id": request.user_id,
        "similar_users": sim_users
    }
