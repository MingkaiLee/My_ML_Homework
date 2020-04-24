import numpy as np
import pandas as pd
from sklearn import linear_model


# 模型构建
solution = linear_model.LinearRegression()
# 数据读取
train_data = pd.read_table('./HomeworkData/prostate_train.txt')
train_data = np.array(train_data)
train_label = train_data[:, 4]
train_data = train_data[:, :4]
test_data = pd.read_table('./HomeworkData/prostate_test.txt')
test_data = np.array(test_data)
test_label = test_data[:, 4]
test_data = test_data[:, :4]
# 线性回归
solution.fit(train_data, train_label)
print('模型参数展示:')
print(solution.coef_)
# 结果预测
forecast = solution.predict(test_data)
print('结果展示:')
print(forecast)
# 求误差平方和
diff = forecast - test_label
res = np.sum(np.square(diff))
print('误差的平方和:')
print(res)
# 求MAE
print('MAE:')
print(np.sum(np.abs(diff))/len(diff))
# 求MAPE
print('MAPE:')
print(np.sum(np.abs(diff)/np.abs(test_label))/len(diff))
# 求MSE
print('MSE:')
print(res/len(diff))
# 求RMSE
print('RMSE:')
print(np.sqrt(res/len(diff)))
# 求R^2
print('R^2:')
print(solution.score(test_data, test_label, sample_weight=None))


"""
引入交叉项
除去布尔型变量项svi
选择用剩下的三个变量组合成交叉项
lcavol*lweight, lweight*lbph, lcavol*lbph
"""
# 构建交叉项数据
train_complex = np.zeros(np.shape(train_data))
test_complex = np.zeros(np.shape(test_data))
train_complex[:, 0] = np.multiply(train_data[:, 0], train_data[:, 1])
train_complex[:, 1] = np.multiply(train_data[:, 0], train_data[:, 2])
train_complex[:, 2] = np.multiply(train_data[:, 1], train_data[:, 2])
train_complex[:, 3] = train_data[:, 3]
test_complex[:, 0] = np.multiply(test_data[:, 0], test_data[:, 1])
test_complex[:, 1] = np.multiply(test_data[:, 0], test_data[:, 2])
test_complex[:, 2] = np.multiply(test_data[:, 1], test_data[:, 2])
test_complex[:, 3] = test_data[:, 3]
# 线性回归
solution.fit(train_complex, train_label)
# 结果预测
pred = solution.predict(test_complex)
print('引入交叉项结果展示:')
print(pred)
# R^2
print('R^2:')
print(solution.score(test_complex, test_label, sample_weight=None))

