#(0927-hw)7107029058_林彥希
#以GD訓練出線性回歸

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#input learning data
click = pd.read_csv("https://raw.githubusercontent.com/wikibook/math-for-ml/master/click.csv", header=None)
print(click)

train = np.loadtxt('https://raw.githubusercontent.com/wikibook/math-for-ml/master/click.csv', delimiter=',', dtype='int', skiprows=1)
df=pd.DataFrame(train)
train_x = train[:,0]
train_y = train[:,1]

plt.scatter(train_x,train_y)
plt.show()

#標準化
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)


#parameter initialization(初始化參數)
theta0 = np.random.rand()
theta1 = np.random.rand()

#predictive function after linear regression(預測功能)
def f(x):
    return theta0 + theta1 * x

#cost function (目標函數)
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

ETA = 1e-3 #learning rate(學習率)
diff = 1 #誤差的差分
count = 0 #更新的回數

#重複參數更新，直到誤差小於0.01
error = E(train_z, train_y)
while diff > 1e-2:

    #將更新結果保存到臨時變量
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    
    theta0 = tmp_theta0 
    theta1 = tmp_theta1

    #計算與上一次會議錯誤的差異
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error

    count += 1
    log = '{}次: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

#圖形
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()