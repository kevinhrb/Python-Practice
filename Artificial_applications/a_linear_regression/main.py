"""
第二版：将原程序向量化
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def data_import(data_name): 
    """
    读取文件,返回列数与数据
    """
    data_path = Path(__file__).parent / "data" / data_name
    
    data = np.loadtxt(data_path, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1:]
    m = x.shape[0]

    return x, y, m

def feature_normalize(x):
    """
    归一化,这里只对x做归一化保留真实预测
    """
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    norm = (x - mu) / sigma
    return norm, mu, sigma

def gradient_descent(x, y, m, alpha=0.1, epsilon=0.01):
    """
    梯度下降法求解线性回归参数(向量版)
    """
    MAX_ITERATION = 1000
    iteration_times = 0
    x_col = x.shape[1]
    theta = np.zeros((1,x_col))
    delta_theta = np.ones((1,x_col))
    loss_history = []
    flag = 0
    while (abs(np.max( abs(delta_theta) ))>epsilon):
        # x:m*n, y:m*1, b:1*1, theta:1*n, y_hat:1*m 
        delta_theta = np.zeros((1,x_col))
        y_hat = np.dot(theta,x.T)
        delta_theta = (1/m) * np.dot((y_hat-y.T),x)

        theta += (-1) * alpha * delta_theta

        J = (1/(2*m)) * np.sum((y_hat-y.T)**2)
        loss_history.append(J)
        iteration_times += 1

        if (iteration_times > MAX_ITERATION):
            flag=1
            break

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iteration times')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

    if flag == 0:
        return 'converged', theta
    elif flag == 1:
        return 'too many times', theta

def verify_function(theta, size, x1, x2, y, verify_num, seed=42):
    """
    随机选几个打印结果,还没改
    """
    if verify_num > size:
        print(f'too many verify_num')
        return

    np.random.seed(seed) # 在choice之前选几个种子
    selected = np.random.choice(size, size=verify_num, replace=False)
    x1_selected = x1[selected]
    x2_selected = x2[selected]
    y_selected = y[selected]
    for i in range(verify_num):
        y_hat = theta[0] + theta[1]*x1_selected[i] + theta[2]*x2_selected[i]
        error = abs((y_selected[i] - y_hat) / y_selected[i])*100
        print(f'y={y_selected[i]:.2f}, predict={y_hat:.2f}, error={error:.4f}%')

def plot(x, y, theta, title):
    """
    自动检测，如果增广矩阵n=3就画三维图
    """

    if x.shape[1] == 3:

        x1 = x[:,1]
        x2 = x[:,2]

        ax = plt.subplot(projection='3d')
        ax.set_title(title)
        
        ax.scatter(x1, x2, y.T, 'r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        a = np.linspace(np.min(x1), np.max(x1), 30)
        c = np.linspace(np.min(x2), np.max(x2), 30)
        X, Y = np.meshgrid(a,c)

        ax.plot_surface(
            X,
            Y,
            theta[0,0] + theta[0,1]*X + theta[0,2]*Y,
            color = 'y',
            alpha = 0.6 # 透明度
        )
        plt.show()
    else:
        print(f'绘图暂时只支持x元素数目为2，目前元素为{x.shape[1]-1}')




def main():
    # 读取数据
    x, y, m = data_import("ex1data2.txt")
    x_norm,_,_ = feature_normalize(x)
    ones = np.ones((m,1))
    x_norm = np.append(ones, x_norm, axis=1) # 增广矩阵

    train_num = int(0.7*m)
    x_train = x_norm[:train_num]
    y_train = y[:train_num]
    x_test = x_norm[train_num:]
    y_test = y[train_num:]

    status, theta = gradient_descent(x_train, y_train, train_num, alpha=0.01)
    print(status, theta)

    # verify_function(theta, m-train_num, x_test, y_test, 5)
    
    plot(x_train, y_train, theta,'train')
    plot(x_test, y_test, theta, 'test')

if __name__ == '__main__':
    main()



