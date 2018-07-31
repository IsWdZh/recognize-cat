# recognize-cat



## 一、使用逻辑回归分类

>逻辑回归简单的实现识别图片中是否有猫



仅仅使用了sigmoid函数逻辑回归构建了一个图片二分类，实现了辨别一张图片中是否有猫。



### 包括

- sigmoid函数

- 初始化参数

- 前向和反向传播

- 梯度下降参数优化

- 预测输入图片

- 整合各模块




### 参数

初始化时，因为仅仅简单使用逻辑回归，所以直接将权重和偏差都初始化为了0。



使用梯度下降来更新参数w和b



### 结果

训练集的预测准确率达到99%

测试集的预测准确率达70%



模型存在过拟合问题



--------------------------------------


## 二、使用神经网络
>在之前使用逻辑回归的基础上，增加了隐含层，使用四层神经网络来进行训练学习，分类图片中是否有猫

###主要包括：
- 初始化参数w和b（每一层的）
- 前向传播
- 计算成本
- 反向传播
- 更新参数

```python
def initialize_parameters_deep(layers_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

### 参数
- 权重参数w随机初始化，再乘上一个很小的值，使参数更加接近y轴
- 偏差b初始化为0


### 结果
![](http://p9huev7ij.bkt.clouddn.com/18-7-31/26362213.jpg)
1. 在学习率为0.0075的情况下迭代2500次，损失函数最终降到0.1以下
2. 模型训练集预测准确率为99%，测试集准确率为78%左右，仍存较大方差
3. 测试集的准确率明显高于使用逻辑回归所预测的结果。
