### 线性回归

#### model

y=WX+b

* 为什么有偏置项？

#### 如何衡量model的好坏

loss function求得Loss(L) : 平方误差

#### 如何求得尽量好的model

gradient descent:

设W为参数，那么求出L对W的导数$\frac{dL}{dW}$

更新W参数$W'=W-\eta\frac{dL}{dW}$

其中$\eta$是预先设定的称为超参数(hyperparameter)

* 我们不想一次性对整个数据集求梯度

所以使用*小批量随机梯度下降*（minibatch stochastic gradient descent）

### 经过代码实现可知

一个神经网络的主要部分有

1. 数据集的加载
2. 基础参数的初始化
3. model的选择
4. loss function的选择
5. 梯度下降方法的选择
6. 超参数的设定
7. 训练过程的编写

## 思考

因为数据是独立同分布的，所以当我们抽出一些数据，并将其拟合的很好，那么如果这组参数是对的话，那么他对其他的数据拟合效果也应该很不错，所以正常情况下，尽管你换了一个batch的数据，loss不应该呈现跳跃式变化。



#### 我想要W里的数据是什么样的



#### 为什么要先对数据进行归一化