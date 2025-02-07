import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    'GPT-4o-0513': [9.3, 13.4, 74.6, 49.9, 32.9, 759],
    'Claude-3.5-Sonnet-1022': [16, 26.7, 78.3, 65, 38.9, 717],
    'o1-mini': [63.6, 80, 90, 60, 53.8, 1820],
    'QwQ-32B-Preview': [44, 60, 90.6, 54.5, 41.9, 1316],
    'DeepSeek-R1-Distill-Qwen-1.5B': [28.9, 52.7, 83.9, 33.8, 16.9, 954],
    'DeepSeek-R1-Distill-Qwen-7B': [55.5, 83.3, 92.8, 49.1, 37.6, 1189],
    'DeepSeek-R1-Distill-Qwen-14B': [69.7, 80, 93.9, 59.1, 53.1, 1481],
    'DeepSeek-R1-Distill-Qwen-32B': [72.6, 83.3, 94.3, 62.1, 57.2, 1691],
    'DeepSeek-R1-Distill-Llama-8B': [50.4, 80, 89.1, 49, 39.6, 1205],
    'DeepSeek-R1-Distill-Llama-70B': [70, 86.7, 94.5, 65.2, 57.5, 1633]
}

# 测试项目
tests = ['AIME 2024 pass@1', 'AIME 2024 cons@64', 'MATH-500 pass@1', 'GPQA Diamond pass@1', 'LiveCodeBench pass@1', 'CodeForces rating']

# 权重
weights = [0.15, 0.15, 0.20, 0.15, 0.15, 0.20]

# 将数据转换为numpy数组
models = list(data.keys())
scores = np.array([data[model] for model in models])

# 计算每个测试项目的最大值，用于归一化
max_scores = scores.max(axis=0)

# 归一化分数
normalized_scores = scores / max_scores

# 计算综合得分
composite_scores = (normalized_scores * weights).sum(axis=1)

# 归一化综合得分
max_composite_score = composite_scores.max()
normalized_composite_scores = composite_scores / max_composite_score

# 添加综合得分到数据中
data['综合分'] = normalized_composite_scores
tests.append('综合分')
scores = np.column_stack((normalized_scores, normalized_composite_scores))

# 设置图形大小
plt.figure(figsize=(18, 10))
# 设置字体
plt.rcParams['font.family'] = ['Microsoft YaHei']

# 绘制柱状图
bar_width = 0.08
index = np.arange(len(tests))

for i, model in enumerate(models):
    plt.bar(index + i * bar_width, scores[i], bar_width, label=model)

# 添加标题和标签
plt.title('柱状图对比')
plt.xlabel('测试项目')
plt.ylabel('分数')
plt.xticks(index + bar_width * (len(models) - 1) / 2, tests, rotation=45)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()