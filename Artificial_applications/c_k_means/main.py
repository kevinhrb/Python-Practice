import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入数据
def data_input(name):
    data_path = Path(__file__).parent / 'data' / name
    x = np.loadtxt(data_path)
    n = len(x[:,0])
    
    return x, n

# kmeans算法
def kmeans(x, mu, K, max_iter=100, tol=1e-6):
    n = len(x)

    for iter in range(1, max_iter+1):
        # 一次性计算每个样本到每个聚类中心的欧氏距离
        # none=np.newaxis，增加一个维度
        # 维度相等或为1，则可以广播
        # （n,1,2）-(1,6,2)->(n,6,2)->distance:(n,6)->在每一行中选最小值为标签
        distance = np.linalg.norm(
            x[:, None, :] - mu[None, :, :],
            axis=2
        )
        x_class = np.argmin(distance, axis=1)

        # 计算聚类中心
        mu_new = mu.copy()
        for k in range(K):
            x_in_class = x[x_class == k] # 掩码
            if len(x_in_class) > 0:
                mu_new[k] = np.mean(x_in_class, axis=0)

        
        if np.linalg.norm(mu_new - mu) < tol:
            mu = mu_new
            break

        mu = mu_new

    # 保证标签与分类一致
    distance = np.linalg.norm(
        x[:, None, :] - mu[None, :, :],
        axis=2
    )
    x_class = np.argmin(distance, axis=1)

    return mu, x_class, iter

# 绘图
def plot(x, x_class, mu):
    plt.figure()
    plt.scatter(x[:,0], x[:,1], c=x_class, cmap='tab10') # cmap=‘tab10’是可用于分类任务的十个不同的颜色，c=label指定类别，cmap指定颜色
    plt.scatter(
        mu[:,0],
        mu[:,1],
        facecolors='none',
        edgecolors='red',
        marker='o',
        label='centre'
    )
    plt.title('k-means scatter')
    plt.legend()
    plt.show()

def main():
    K = 6
    x,n = data_input('ch6_kmeans_data1.txt')
    np.random.seed(42)
    select_idx = np.random.choice(n, K, replace=False)
    mu = x[select_idx]
    mu_final, label, iter= kmeans(x, mu, K)
    print('聚类中心 mu: \n', mu_final)
    print(iter)
    plot(x, label, mu_final)

if __name__ == '__main__':
    main()


