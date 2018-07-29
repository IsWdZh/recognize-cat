import numpy as np
# import h5py
from matplotlib import pyplot as plt
from data_utils import load_dataset


train_set_x_org, train_set_y, test_set_x_org, test_set_y, classes = load_dataset()

# plt.imshow(train_set_x_org[10])
# plt.show()

m_train = train_set_x_org.shape[0]      # 训练用例个数
m_test = test_set_x_org.shape[0]
num_px = train_set_x_org.shape[1]       # 每个图片像素

train_set_x_flatten = train_set_x_org.reshape(train_set_x_org.shape[0],-1).T    # 将每张图片转换为列向量，重塑矩阵为（nums_px * nums_px * 个数）
test_set_x_flatten = test_set_x_org.reshape(test_set_x_org.shape[0],-1).T
# print(train_set_x_flatten.shape)

train_set_x = train_set_x_flatten/255   # 标准化数据集，将每个用例的整个numpy矩阵平均结构化
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_parameters(dim):
    '''
    :param dim: w向量的大小，与每个输入的列向量元素相等
    :return:
    '''
    w = np.zeros((dim,1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    '''
    前向传播和反向传播
    :param w: numpy矩阵的大小（num_px * numpx * 3, 1）
    :param b: 偏差（实数）
    :param X: 数据的大小规模（num_px * numpx * 3, 用例数目）
    :param Y: 真实标签向量尺寸（1，用例的数量）
    :return: 反向传播的dw，db用于更新参数，以及损失的成本
    '''
    m = X.shape[0]     # 训练用例个数，做平均用

    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)

    cost = np.squeeze(cost)       # 将向量转换为矩阵

    gradients = {"dw":dw,
                 "db":db}
    return gradients, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    优化，根据设置的迭代次数，不断优化参数。
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 迭代过程中是否输出代价函数值
    :return: parameters，gradients，costs
    '''
    costs = []
    for i in range(num_iterations):
        gradients, cost = propagate(w, b , X, Y)

        dw = gradients["dw"]
        db = gradients["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("经过迭代 %d 次后，代价函数cost = %f" %(i, cost))
    parameters = {"w":w,
                  "b":b}
    gradients = {"dw":dw,
                 "db":db}
    return parameters, gradients, costs


def predict(w, b ,X):
    '''
    预测输入的图片
    :return: 0 or 1
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))   # 初始化创建一个全0预测标签
    w = w.reshape(X.shape[0], 1)   # 重塑矩阵，保证和输入特征向量相同
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:                  # 做出预测，小于0.5概率的就认为不存在
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def whole_model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    w, b = initialize_parameters(X_train.shape[0])   # 提取一张图片 一个列向量元素个数（12288）
    parameters, gradients, costs = optimize(w, b ,X_train, Y_train, num_iterations, learning_rate, print_cost = True)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b ,X_train)

    print("训练准确率：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试准确率：{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs":costs,
         "Y_prediction_test":Y_prediction_test,
         "Y_prediction_train":Y_prediction_train,
         "w":w,
         "b":b,
         "learning_rate":learning_rate,
         "num_iterations":num_iterations}
    return d

d = whole_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.3, print_cost = True)

costs = np.squeeze(d["costs"])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(costs)
plt.ylabel('损失成本')
plt.xlabel('迭代次数（每百次）')
plt.title('学习率 = ' + str(d["learning_rate"]))
plt.show()

# learning_rates = [0.25, 0.3, 0.35]
# models = {}
# for i in learning_rates:
#     print("learning_rates is: "+ str(i))
#     models[str(i)] = whole_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=i, print_cost = True)
#     print('\n'+'--------------------------------------------------------'+'\n')
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.ylabel('损失成本')
# plt.xlabel('迭代次数（每百次）')
# legend = plt.legend(loc='upper center', shadow = True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()


# my_image = "1.jpg"
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
#
# plt.imshow(image)
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + \
#       classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

