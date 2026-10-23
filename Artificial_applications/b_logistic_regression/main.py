"""
逻辑回归，用于非线性拟合
修改状态变量[x1,x2,x1^2,x2^2]
"""
import numpy as np
import math
from pathlib import Path

def data_import(data_name):
    """
    读文件,保证数据保存在data文件夹中
    """
    data_path = Path(__file__).parent / 'data' / data_name
    x1 = np.loadtxt(data_path, usecols=0, delimiter=',')
    x2 = np.loadtxt(data_path, usecols=1, delimiter=',')
    y = np.loadtxt(data_path, usecols=2, delimiter=',')
    m = len(x1)

    return x1, x2, y, m

def gradient_descent(x1, x2, y, split, alpha=0.1, epsilon=0.01):
    MAX_ITERATION = 1000
    iteration_times = 0
    theta = np.zeros(3)
    while True:
        delta_theta = np.zeros(3) # 每次迭代清零梯度！！
    
        for i in range(split):
            z = theta[0]+theta[1]*x1[i]+theta[2]*x2[i]
            y_hat = 1 / (1+np.exp(-z)) # sigmoid修正
            delta_theta[0] += (y_hat - y[i])
            delta_theta[1] += (y_hat - y[i]) * x1[i]
            delta_theta[2] += (y_hat - y[i]) * x2[i]
        # 更新theta
        theta[0] -= (1/split) * alpha * delta_theta[0]
        theta[1] -= (1/split) * alpha * delta_theta[1]
        theta[2] -= (1/split) * alpha * delta_theta[2]
        iteration_times += 1
        # 收敛判断
        if np.max(np.abs(delta_theta)) < epsilon:
            return 'converged : 普通梯度下降', theta
        if iteration_times > MAX_ITERATION:
            return 'too much times', theta


def feature_normalization(x):
    mu = np.mean(x)
    sigma = np.std(x)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

def acc(theta, x1, x2, y, verify_num, name,seed=42):
    size = len(x1)
    if verify_num > size :
        return 'too much verify_num'
    np.random.seed(seed)
    selected = np.random.choice(size, size=verify_num, replace=False)
    x1_selected = x1[selected]
    x2_selected = x2[selected]
    y_selected = y[selected]
    acc_num = 0
    total_num = len(y_selected)
    for i in range(total_num):
        z = theta[0]+theta[1]*x1_selected[i]+theta[2]*x2_selected[i]
        y_hat = 1 / (1+np.exp(-z))
        classify = 1 if y_hat>0.5 else 0
        if abs(classify - y_selected[i]) < 1e-3:
            acc_num +=1
    acc_rate = acc_num / total_num *100
    print(f'acc_rate_{name}={acc_rate:.4f}%')

        

def main():
    data_name =  'ex2data2.txt'
    x1, x2, y, m = data_import(data_name)
    x1, _, _ = feature_normalization(x1)
    x2, _, _ = feature_normalization(x2)
    
    split = int(0.7*m)

    x1_train = x1[:split]
    x2_train = x2[:split]
    y_train = y[:split]
    x1_test = x1[split:]
    x2_test = x2[split:]
    y_test = y[split:]

    situ, theta = gradient_descent(x1_train, x2_train, y_train, split, alpha=0.1, epsilon=0.01)
    print(f'situ:{situ},theta:{theta}')
    acc(theta, x1_train, x2_train, y_train, split, 'train')
    acc(theta, x1_test, x2_test, y_test, m-split, 'test')

if __name__ == '__main__':
    main()