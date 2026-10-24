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
    
    data = np.loadtxt(data_path)
    x = data[:, :-1]
    y = data[:, -1:]
    m = x.shape[0]

    return x, y, m

def feature_normalize(x):
    """
    归一化
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

    if flag == 0:
        return 'converged', theta, loss_history
    elif flag == 1:
        return 'too many times', theta, loss_history

def plot_loss_curves(loss_histories):
    plt.figure()
    for alpha, loss_history in loss_histories.items():
        plt.plot(loss_history, label=f'learning rate={alpha}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves for different learning rates')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(x, y, title, x_label, color='r'):
    ax = plt.subplot()
    ax.set_title(title)
    ax.scatter(x, y, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel('reach envelope')
    plt.show()

def plot_prediction(y_true, y_pred):
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - residual_sum / total_sum
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))

    plt.figure()
    plt.scatter(y_true, y_pred, label='test sample')
    plt.plot([low, high], [low, high], label='ideal prediction line')
    plt.xlabel('real reach envelope / cm')
    plt.ylabel('predict reach envelope / cm')
    plt.title(f'predict consequence')
    plt.legend()
    plt.grid(False)
    plt.axis('equal')
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.show()




def main():
    # 读取数据
    x, y, m = data_import("reach_data_120.txt")
    plot_scatter(x[:,0],y,'scatter plot of shoulder width versus reach encelope','shoulder width')
    plot_scatter(x[:,1],y,'scatter plot of height versus reach envelop','height', color='b')

    x_norm,mu,sigma = feature_normalize(x)
    y_norm,mu_y,sigma_y = feature_normalize(y)
    print(f'肩宽均值：{mu[0]:.4f}, 身高均值：{mu[1]:.4f},可达域均值：{mu_y[0]:.4f}')

    # 增广矩阵，不单独写b
    ones = np.ones((m,1))
    x_norm = np.append(ones, x_norm, axis=1) # 增广矩阵

    # 划分数据集
    train_num = int(0.7*m)
    x_train = x_norm[:train_num]
    y_train = y_norm[:train_num]
    x_test = x_norm[train_num:]

    learning_rates = [0.5, 0.1, 0.05, 0.01, 0.001]
    loss_histories = {}
    models = {}
    for alpha in learning_rates:
        models[alpha] = gradient_descent(
            x_train, y_train, train_num, alpha=alpha
        )
        loss_histories[alpha] = models[alpha][2]
    plot_loss_curves(loss_histories)

    alpha = 0.05
    status, theta, _ = models[alpha]
    theta_real = mu_y/mu[0] * theta
    print(f'learning rate={alpha}, {status}, {theta_real}')

    # 使用测试集预测可达距离，并还原为 cm
    y_pred = (theta @ x_test.T).T * sigma_y + mu_y
    y_test_real = y[train_num:]
    errors = y_pred[:, 0] - y_test_real[:, 0]
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    print(f'MAE={mae:.4f} cm, MSE={mse:.4f} cm^2')
    print('测试集预测结果：')
    print(f"{'样本':>4} {'肩宽(cm)':>10} {'身高(cm)':>10} {'预测可达域(cm)':>16} {'真实可达域(cm)':>16} {'err(cm)':>10}")
    for i, (size, predicted, actual, error) in enumerate(
        zip(x[train_num:], y_pred[:, 0], y_test_real[:, 0], errors), start=1
    ):
        print(
            f'{i:>4} {size[0]:>10.1f} {size[1]:>10.1f} '
            f'{predicted:>16.2f} {actual:>16.2f} {error:>10.2f}'
        )

    plot_prediction(y_test_real[:, 0], y_pred[:, 0])

if __name__ == '__main__':
    main()



