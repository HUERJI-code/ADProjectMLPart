使用步骤：
1、conda install torch，numpy，pandas，fastapi，scikit-learn
2、每当需要更新模型或者首次训练模型时：run python app/trainmodel.py
3、需要测试模型但不需要使用API时,run app/recommender.py，注意自行在代码中更改推荐top_k个数据等
4、需要调用API时，需要 run uvicorn app.main:app --reload --port 8000
5、API传输json例子：
	请求推荐活动（只返回 ID）：
	POST http://localhost:8000/recommend/
	{
	  "user_id": 1,
	  "top_k": 2
	}
	返回数据为：JSON{
		"user_id": request.user_id, 
		"recommended_activity_ids": recommendations#['id', 'title', 'score']
	}
	查找相似用户：
	POST http://localhost:8000/similar-users/
	{
	  "user_id": 1,
	  "top_k": 3
	}
	返回数据为：JSON{
		"user_id": request.user_id,
        "similar_users": [
            {"user_id": uid, "similarity": score} for uid, score in sim_users
        ]
	}
6、补充说明：我在python端自动加入特征矩阵的更新判断为：每有一个新用户创建或每有10个新活动创建，可自行修改，如果不想在python判断，你可以自行在C#后端加入

接下来是使用数据的csv格式：
activity.csv：
id,title,description,starttime,endtime,content,maxNum,tag

useractivityevaluations.csv
userid,activityid,favorite
其中favorite的值为1：用户收藏；2：用户已参加（这里的已参加是这个活动已经举办过了，相当于作为历史参与活动）

接下来是模型的粗略介绍：
预处理部分：
1、首先将每个活动的信息按类型处理整合成一个特征向量：该向量由三部分组成：文本向量（300维）、标签向量（多标签独热编码）、时间特征（3维：hour_sin, hour_cos, duration）
2、然后对用户标签处理：经过多标签编码后构成一个标签索引序列，标签索引序列经过嵌入层和平均池化后形成稠密标签表示
3、将用户id和操作2得到的稠密标签进行嵌入，形成用户特征矩阵
3、将经过一个全连接层处理后的1中的特征向量和对应的活动id拼接起来，形成活动特征矩阵
4、将3中的用户特征矩阵和4中的活动特征矩阵拼接起来
模型预测部分：
将预处理后的混合特征矩阵交给深度协同过滤网络：由三层MLP构成，处理后进行最终预测

训练阶段：使用交互行为（收藏/参加）作为监督信号训练模型
推荐阶段：过滤掉已交互的活动，仅对新活动进行评分与排序

