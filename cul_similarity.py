import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')  # 更改为 TkAgg 后端，适用于大多数情况


# 假设你的CSV文件路径分别为'poison_samples.csv' 和 'normal_samples.csv'
poison_file_path = r'poisoned_samples_ps30_ws8.csv'#中毒良性样本 
normal_file_path = r'original_samples_ps30_ws8.csv'#干净良性样本

# 1. 读取CSV文件
# 假设CSV文件的前两列是非特征数据，从第三列开始是特征
poison_data = pd.read_csv(poison_file_path)
normal_data = pd.read_csv(normal_file_path)

# 随机抽样

#poison_data_sampled = poison_data.sample(n=100, random_state=42)  # 随机抽取20条数据
normal_data_sampled = normal_data.sample(n=10000, random_state=42)  # 随机抽取10000条数据

# 提取特征数据（从第三列开始）
poison_samples = poison_data.iloc[:, 1:].values  # 投毒样本的特征
normal_samples = normal_data_sampled.iloc[:, 1:].values  # 正常样本的特征

# 2. 计算投毒样本与正常样本的余弦相似度
# cdist用于计算两个集合之间的所有样本点的相似度（注意使用'cosine'度量）
similarities = 1 - cdist(poison_samples, normal_samples, metric='cosine')

# 对于每个投毒样本，选择它与所有正常样本的最大相似度（即它最相似的正常样本）
max_similarities = np.max(similarities, axis=1)

# 3. 计算相似度的均值、标准差、最大值和最小值
mean_similarity = np.mean(max_similarities)
std_similarity = np.std(max_similarities)
max_similarity = np.max(max_similarities)
min_similarity = np.min(max_similarities)

# 打印统计结果
print(f"投毒样本与正常样本的最大余弦相似度统计:")
print(f"均值: {mean_similarity:.4f}")
print(f"标准差: {std_similarity:.4f}")
print(f"最大值: {max_similarity:.4f}")
print(f"最小值: {min_similarity:.4f}")