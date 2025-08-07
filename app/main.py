# app/main.py

# python -m uvicorn app.main:app --reload --port 8000

import os
import subprocess
from fastapi import FastAPI, BackgroundTasks
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

def _run_retraining():
    """
    在子进程中调用 trainmodel 脚本重新训练并保存模型。
    注意：请确保你的 trainmodel.py
    顶层模块名是 app.trainmodel，且可以使用 `python -m app.trainmodel` 方式运行。
    """
    # 切到项目根目录，避免路径问题
    cwd = os.getcwd()
    # 启动子进程执行重训脚本
    subprocess.run(
        ["python", "-m", "app.trainmodel"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

@app.get("/retrain/")
def retrain(background_tasks: BackgroundTasks):
    """
    异步触发模型重训练。立即返回，由后台任务去执行 trainmodel.py。
    """
    background_tasks.add_task(_run_retraining)
    return {"message": "Model retraining has been started in background."}