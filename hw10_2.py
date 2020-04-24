import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso


model_linear = LinearRegression()
model_ridge = Ridge(alpha=1)
model_lasso = Lasso(alpha=1)


# 测试拟合参数稳定性的函数
def stability_test(echo):
    for e in range(0, echo):
        x1 = np.linspace(1, 20, 20)
        x2 = np.zeros((20, 1))
        y = np.zeros((20, 1))
        # 生成y和特征x2
        for i in range(0, 20):
            y[i] = 3 * x1[i] + 2 + np.random.normal(0, np.sqrt(2))
            x2[i] = 0.05 * x1[i] + np.random.normal(0, np.sqrt(0.5))
        x1 = x1.reshape(-1, 1)
        # 合并输入特征
        X = np.hstack((x1, x2))
        model_linear.fit(X, y)
        model_ridge.fit(X, y)
        model_lasso.fit(X, y)
        print('echo{}:'.format(e))
        print('Linear:{}'.format(model_linear.coef_))
        print('Ridge:{}'.format(model_ridge.coef_))
        print('Lasso:{}'.format(model_lasso.coef_))


# 正则化系数改变
def alpha_test():
    x1 = np.linspace(1, 20, 20)
    x2 = np.zeros((20, 1))
    y = np.zeros((20, 1))
    # 生成y和特征x2
    for i in range(0, 20):
        y[i] = 3 * x1[i] + 2 + np.random.normal(0, np.sqrt(2))
        x2[i] = 0.05 * x1[i] + np.random.normal(0, np.sqrt(0.5))
    x1 = x1.reshape(-1, 1)
    # 合并输入特征
    X = np.hstack((x1, x2))
    for i in range(0, 10):
        model = Ridge(alpha=i)
        model.fit(X, y)
        print('alpha={}'.format(i))
        print('Ridge:{}'.format(model.coef_))
        model = Lasso(alpha=i)
        model.fit(X, y)
        print('Lasso:{}'.format(model.coef_))


if __name__ == '__main__':
    # stability_test(10)
    alpha_test()
