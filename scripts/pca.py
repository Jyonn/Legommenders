import os

import numpy as np
from sklearn.decomposition import PCA


embedding_path = os.path.join(
    '../saving',
    f'MIND-Distillation-All',
    'NAML',
    'free-train_get_user_embedding',
    'user_embeddings.npy'
)


# 加载.npy文件
data = np.load(embedding_path)  # 替换成你的.npy文件路径

# 创建PCA模型，将维度从64维降低到32维
pca = PCA(n_components=32)

# 使用PCA模型拟合数据并进行降维
data_reduced = pca.fit_transform(data)

# data_reduced现在包含了降维后的数据，每个样本有32维

# 如果需要，你可以保存降维后的数据为.npy文件
np.save(embedding_path.replace('.npy', '.d32.npy'), data_reduced)
