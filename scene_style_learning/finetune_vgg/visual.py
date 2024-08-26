import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap

# 设置文件路径
feature_dir = '/data0/JM/code/scene_style_learning/finetune_vgg/feature_std'
npy_files = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.npy')])

# 读取 .npy 文件并加载数据
features = []
for file in tqdm(npy_files[:10000], desc="Loading .npy files"):  # 先用前10000个文件测试
    feature = np.load(file).reshape(512)  # 将 (1, 512) 变为 (512)
    features.append(feature)

# 将特征列表转换为 numpy 数组
features = np.vstack(features)

# 检查数据中是否有NaN或无穷大值
print(f"NaN values: {np.isnan(features).sum()}")
print(f"Infinity values: {np.isinf(features).sum()}")

# 替换NaN和无穷大值
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# 再次检查数据中是否有NaN或无穷大值
print(f"NaN values after replacement: {np.isnan(features).sum()}")
print(f"Infinity values after replacement: {np.isinf(features).sum()}")

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用PCA将数据预降维到50维
pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(features_scaled)

# 使用UMAP将数据降维到2D
reducer = umap.UMAP(n_components=2, random_state=42)
features_2d = reducer.fit_transform(features_pca)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], s=6, cmap='Spectral')
plt.title('UMAP Visualization of Features')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.colorbar()
plt.savefig('temp.png')
