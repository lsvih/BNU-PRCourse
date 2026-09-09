"""
Author: Yanzeng Li
Date: 2026-09-09
Description: 手写 K-Means 聚类并可视化迭代过程（簇划分、质心移动与 SSE 下降）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

np.random.seed(42)

# 生成 3 个高斯簇的数据
n_samples = 300
k = 3
X, _ = make_blobs(n_samples=n_samples, centers=k, cluster_std=1.2, random_state=59)

# 随机选择 k 个数据点作为初始质心（Forgy 初始化）
init_idx = np.random.choice(n_samples, k, replace=False)
centroids = X[init_idx].copy()


def assign_and_update(X, centroids):
    """一轮 K-Means 迭代：先分配，再更新质心，返回标签、新质心和 SSE"""
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    sse = np.sum(np.min(dists, axis=1) ** 2)  # 当前划分的平方误差和

    new_centroids = centroids.copy()
    for i in range(k):
        mask = labels == i
        if mask.any():                        # 空簇则保持原质心不变
            new_centroids[i] = X[mask].mean(axis=0)
    return labels, new_centroids, sse


# 迭代运行 K-Means，记录每一轮的状态
max_iter = 50
tol = 1e-4

labels_history, centroids_history, sse_history = [], [], []

for i in range(max_iter):
    labels, new_centroids, sse = assign_and_update(X, centroids)

    labels_history.append(labels.copy())
    centroids_history.append(centroids.copy())
    sse_history.append(sse)

    if np.allclose(centroids, new_centroids, atol=tol):
        break
    centroids = new_centroids

n_iter = len(sse_history)
centroids_history = np.array(centroids_history)
final_centroids = centroids_history[-1]

print(f"K-Means 迭代 {n_iter} 轮收敛")
print(f"最终质心：\n{final_centroids}")
print(f"最终 SSE（惯性）：{sse_history[-1]:.2f}")

# 与 sklearn 的结果对比
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
print(f"sklearn KMeans 惯性：{km.inertia_:.2f}")


"""
后面全部都是绘图和动画了，同学可以无视
"""
# 数据范围与绘图网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))


def grid_labels(centers):
    """计算绘图网格上每个点属于哪个簇（用于画 Voronoi 区域）"""
    pts = np.c_[xx.ravel(), yy.ravel()]
    d = np.linalg.norm(pts[:, None] - centers[None, :], axis=2)
    return np.argmin(d, axis=1).reshape(xx.shape)


fig = plt.figure(figsize=(15, 5))

# 图1: 数据点、当前簇划分与质心
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_title('K-Means Clustering')
ax1.grid(True)

# 图2: 损失函数曲面（固定其余质心在最终位置，只移动第 1 个质心）
gx = np.linspace(x_min, x_max, 50)
gy = np.linspace(y_min, y_max, 50)
GX, GY = np.meshgrid(gx, gy)
loss = np.zeros_like(GX)

for i in range(GX.shape[0]):
    for j in range(GX.shape[1]):
        c = final_centroids.copy()
        c[0] = [GX[i, j], GY[i, j]]
        d = np.linalg.norm(X[:, None] - c[None, :], axis=2)
        loss[i, j] = np.sum(np.min(d, axis=1) ** 2)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(GX, GY, loss, cmap='viridis', alpha=0.7, rstride=2, cstride=2)
path = ax2.plot([], [], [], 'r-', linewidth=2)[0]
point = ax2.plot([], [], [], 'ro')[0]
ax2.set_xlabel('x0')
ax2.set_ylabel('y0')
ax2.set_zlabel('Loss')
ax2.set_title('Loss Function Surface')
ax2.view_init(elev=30, azim=45)

# 图3: SSE 变化曲线
ax3 = fig.add_subplot(1, 3, 3)
sse_line, = ax3.plot([], [], 'b-', linewidth=2, label='SSE (Inertia)')
current_point = ax3.plot([], [], 'ro')[0]
ax3.set_xlim(0, n_iter - 1)
ax3.set_ylim(0, max(sse_history) * 1.1)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('SSE')
ax3.set_title('Loss Curve')
ax3.grid(True)
ax3.legend()

plt.tight_layout()

# 第 1 个质心在曲面上的运动轨迹
path_coords = []
for c0 in centroids_history[:, 0]:
    x_idx = np.argmin(np.abs(gx - c0[0]))
    y_idx = np.argmin(np.abs(gy - c0[1]))
    path_coords.append((c0[0], c0[1], loss[y_idx, x_idx]))
path_coords = np.array(path_coords).T


def update(frame):
    labels = labels_history[frame]
    centers = centroids_history[frame]

    # 图1：重绘当前簇划分
    ax1.clear()
    ax1.contourf(xx, yy, grid_labels(centers),
                 levels=np.arange(-0.5, k, 1), cmap='viridis', alpha=0.25)
    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                vmin=0, vmax=k - 1, edgecolors='k', label='Data points')
    for i in range(k):  # 质心移动轨迹
        traj = centroids_history[:frame + 1, i]
        ax1.plot(traj[:, 0], traj[:, 1], 'r--', linewidth=1, alpha=0.6)
    ax1.scatter(centers[:, 0], centers[:, 1], marker='*', s=300,
                c='red', edgecolors='k', label='Centroids')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f'Iteration {frame}: SSE = {sse_history[frame]:.1f}')
    ax1.grid(True)
    ax1.legend()

    # 图2：更新曲面上的轨迹
    path.set_data(path_coords[0, :frame + 1], path_coords[1, :frame + 1])
    path.set_3d_properties(path_coords[2, :frame + 1])
    point.set_data([path_coords[0, frame]], [path_coords[1, frame]])
    point.set_3d_properties([path_coords[2, frame]])

    # 图3：更新 SSE 曲线
    sse_line.set_data(range(frame + 1), sse_history[:frame + 1])
    current_point.set_data([frame], [sse_history[frame]])

    return path, point, sse_line, current_point


ani = FuncAnimation(
    fig,
    update,
    frames=range(n_iter),
    interval=1500,
    blit=False,
    cache_frame_data=False,
    repeat=True,
    repeat_delay=2000
)

plt.ion()
plt.show(block=True)

while True:
    plt.pause(0.1)
