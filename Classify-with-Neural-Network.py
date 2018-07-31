import numpy as np
# import h5py
from matplotlib import pyplot as plt
from data_utils import load_dataset
import scipy
from scipy import ndimage



train_x_org, train_y, test_x_org, test_y, classes = load_dataset()

# plt.imshow(train_x_org[10])
# plt.show()

m_train = train_x_org.shape[0]      # 训练用例个数
m_test = test_x_org.shape[0]
num_px = train_x_org.shape[1]       # 每个图片像素

train_x_flatten = train_x_org.reshape(train_x_org.shape[0],-1).T    # 将每张图片转换为列向量，重塑矩阵为（nums_px * nums_px * 个数）
test_x_flatten = test_x_org.reshape(test_x_org.shape[0],-1).T     # "-1"使剩余的尺寸重塑为平的
# print(train_x_flatten.shape)

train_x = train_x_flatten/255   # 标准化数据集，将每个用例的整个numpy矩阵平均结构化
test_x = test_x_flatten/255


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0                     # Z小于等于0即令其梯度为0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def initialize_parameters_deep(layer_dims):
    '''
    初始化深层神经网络参数
    :param layer_dims: 各层神经元list，包含每层的特征数.
                        如一个两层的神经网络，则len(layer_dims) = 3 ,分别为[输入特征数，隐藏层，输出层]
    :return: 参数字典，包括各层的权重W和偏差b
                W -- (layer_dims[l], layer_dims[l - 1])
                b -- (layer_dims[l], 1)
    '''
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01     没搞懂为什么要除以上一层开平方，但效果确实好
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    '''
    前向传播线性部分
    :param A:  前一层激活值
    :param W:  权重
    :param b:  偏差
    :return:  Z -- 经过线性变换的激活前参数
               cache -- 字典，存储前向传播过程中的所有值
    '''
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    '''
    前向传播
    :param A_prev: 前一层(或输入层)
    :param W: （当前层尺寸，前一层尺寸）
    :param b: （当前层尺寸，1）
    :param activation: 选择激活函数
    :return: A -- 经过一次前向传播的激活值
              cache -- 存储前向传播过程中的值，用于反向传播计算使用。
    '''
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    '''
    L层神经网络前向传播 Relu*(L-1)  ->   Sigmoid
    :param X:  输入数据（输入的数据尺寸，图片例子的数量）
    :param parameters: 初始化得到的参数initialize_parameters_deep()
    :return: AL -- 激活值
              cache -- 每一次前向传播激活函数后缓存的 （A, W, b, Z)
    '''
    caches = []
    A = X
    L = len(parameters) // 2            # 网络的层数

    # 前面 L-1次使用relu激活函数传递，保存传播过程的值
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation='relu')
        caches.append(cache)

    # 最后一次使用 sigmoid 函数，做二分类
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    '''
    计算损失
    :param AL: 最后的预测输出，形状为（1，图片用例数量）
    :param Y:  真实标签向量（0-没有，1-有） 形状为（1，图片用例数量）
    :return:  交叉熵损失
    '''
    m = Y.shape[1]
    # cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = -1/m * np.sum(np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log((1-AL).T)))

    cost = np.squeeze(cost)  # 确保计算得到形状是对的
    return cost


def linear_backward(dZ, cache):
    '''
    单层的反向传播线性部分
    :param dZ:  当前层的关于线性计算结果的梯度
    :param cache:  当前层前向传播过程中存储的值，元组（A_prev, W， b）
    :return:  dA -- 损失函数关于前一层的梯度，与A_prev形状相同
               dW -- 对W的梯度，与W形状相同
               db -- 对b的梯度，与b形状相同
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    '''
    反向传播过程 激活函数部分
    :param dA: 当前层激活值的梯度
    :param cache: 元组（Z，激活缓存（A， W， b））
    :param activation: 选择激活函数
    :return:   dA_prev -- 上一层激活值的梯度
                dW -- 交叉熵损失对W的梯度
                db -- 交叉熵损失对b的梯度
    '''
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    '''
    反向传播过程，前 L-1 层使用Relu激活函数，最后使用Sigmoid激活函数
    :param AL: 前向传播预测的输出
    :param Y: 真是标签值
    :param caches:     'Relu'前向传播的所有缓存值[caches[i] for i in range(L-1)]
                        'Sigmoid'传播的缓存 (caches[L-1])
    :return: grads -- 各层参数的梯度
    '''
    grads = {}
    L = len(caches)  # 神经网络层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # 将真是标签值转换为和预测输出向量同样的形状

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))     # 从最后的输出值开始反向计算梯度

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] =  \
                                                        linear_activation_backward(dAL, current_cache, "sigmoid")

    # 反向传播，从最后一层开始往前推
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    '''
    更新参数
    :param parameters: 参数
    :param grads: 梯度
    :param learning_rate: 学习率
    :return: 更新后各层的参数
    '''
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    ### END CODE HERE ###
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    '''
    L层神经网络， 前L-1层为Relu激活函数，最后使用Sigmoid激活函数
    :param X: 输入数据，形状为（图片例子的数目，num_px * num_px * 3）
    :param Y: 真是标签向量
    :param layers_dims: 从输入到中间层到输出每层的神经元数，长度为网络层数+1
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否每100次输出依次损失值
    :return:  parameters -- 模型学习后的参数，可用于预测
    '''
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        # 前向传播一次，反向计算梯度一次
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    print(np.squeeze(costs))
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, y, parameters):
    '''
    用于检测计算训练集和测试集的预测精度。（需要有真实标签）
    :param X: 输入数据
    :param y: 标签
    :param parameters: 参数
    :return:  p -- 对于输入X的预测值
    '''
    m = X.shape[1]
    n = len(parameters) // 2     # 神经网络层数
    p = np.zeros((1,m))

    # 前向传播
    probas, caches = L_model_forward(X, parameters)

    # 将激活值probas转换为0或1的预测值
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("预测准确率为：" + str(np.sum((p == y)/m)))
    return p


layers_dims = [12288, 20, 7, 5, 1]      #  4-layer
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# print(parameters)
# pred_train = predict(train_x, train_y, parameters)    # 训练集准确率
# pred_test = predict(test_x, test_y, parameters)


# 使用其他的图片
my_image = "cat.jpg"
my_label_y = [1]       # 该图片应该属于的分类

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))  # 读取cat.jpg图片

# 改变图片的尺寸形状
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", 模型预测这是一只 \"" + \
       classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" 的图片。")


# print(type(classes[1]))     numpy.bytes
# print(classes[1].decode("utf-8"))       utf-8 cat


