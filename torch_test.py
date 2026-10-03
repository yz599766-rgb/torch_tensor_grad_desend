#一、张量 Tensor：创建
import torch

# 1. 直接从列表创建张量
t1 = torch.tensor([1,2,3])
print("t1 1维张量：", t1)
print("形状shape：", t1.shape)

# 2. 2维张量（矩阵）
t2 = torch.tensor([[1,2],[3,4]])
print("\nt2 2维张量：\n", t2)
print("t2 shape：", t2.shape)

# 3. 创建全0、全1张量，常用初始化权重
zeros = torch.zeros(2,3)  # 2行3列全0
ones = torch.ones(2,3)
print("\n全0张量：\n", zeros)
print("\n全1张量：\n", ones)

# 4. 创建随机数张量（神经网络初始化最爱用）
rand_t = torch.rand(2,3) # 0~1之间均匀随机数
randn_t = torch.randn(2,3) # 标准正态分布（均值0方差1） 生成张量，张量里的数字服从【标准正态分布】
print("\n随机张量rand：\n", rand_t)
print("\n随机张量randn_t：\n", randn_t)

# 5. 从numpy数组转张量（经常用到）
import numpy as np
arr = np.array([1,2,3])                      #发现只要是张量打印的时候前面就会加一个tensor
print(arr)
t_from_np = torch.from_numpy(arr)
print("\nnumpy转tensor：", t_from_np)

#二、张量运算（大白话）

# import torch
# #创建两个**一维张量 a 和 b**，数据分别是 `[1,2,3]` 和 `[4,5,6]`，并且指定里面数字的数据类型为**32 位浮点数**。
# a = torch.tensor([1,2,3], dtype=torch.float32)  #张量运算尽量**同数据类型**，整数张量做求导会报错，一般用`float32`
# b = torch.tensor([4,5,6], dtype=torch.float32)

# # 1. 逐元素加减乘除（不是矩阵乘法！）
# print("a + b =", a + b)
# print("a * b =", a * b) # 对应位置相乘：1*4,2*5,3*6

# # 2. 矩阵乘法（2种写法，推荐@）
# A = torch.tensor([[1,2],[3,4]], dtype=torch.float32)  #两行两列矩阵
# B = torch.tensor([[1],[2]], dtype=torch.float32)      #两行一列矩阵
# C = A @ B
# print("\n矩阵乘法 A@B：\n", C)

# # 3. 广播机制（和numpy一样，小的自动扩展形状做运算）     如何扩展：重复多几层变成同行
# x = torch.tensor([[1],[2],[3]])
# y = torch.tensor([10,20])
# print("\n广播结果：\n", x + y)


# # 三、GPU 张量：把张量放到显卡跑
# import torch

# # 查看是否有可用GPU 自动判断你的电脑**有没有可用 NVIDIA 显卡（GPU）    有 GPU，就选用 `cuda`（显卡）作为计算设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("当前设备：", device)

# # 方法1：创建张量直接指定设备
# t_gpu = torch.tensor([1,2,3], device=device)
# # 方法2：已有张量转到GPU
# t_cpu = torch.tensor([4,5,6])
# t_gpu2 = t_cpu.to(device)

# # GPU张量转回CPU（画图、转numpy必须先回CPU！numpy不认识GPU张量）
# t_back_cpu = t_gpu2.cpu()
# print("转回cpu：", t_back_cpu)


# ## 四、`requires_grad` 开启梯度记录【Autograd 核心】`requires_grad=True` 开启 Autograd 记账本，记录正向计算图。

# > 
# > 大白话：
# > `requires_grad=False`（默认）：只是单纯算数字，**不记录计算过程，不求导**
# > `requires_grad=True`：PyTorch 开启 “记账本”，记录这个张量后续所有运算，生成计算图,最后能自动求梯度。


import torch

# w是我们要训练的参数，打开requires_grad ：告诉 PyTorch：**盯紧这个变量！所有跟 w 相关的计算全部记录下来，后面要自动求导**。
# w：神经网络里的**权重 weight**，属于模型参数，我们要不断更新它。x：**输入数据**，就是样本，固定不变，不需要求导。
#2.0  3.0为初始值写`2`是整数，**整数张量不能开`requires_grad=True`做自动求导**；`2.0`是浮点数，才能用来做梯度计算。
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0) # 输入，不需要求导，默认False



y = w * x  # 计算 y = w*x
print(y)
# y这个张量，自带grad_fn，代表它是通过运算生成的，可以反向求导
print(y.grad_fn)             #这里的grad_fn  就是Mulbackward0
#输出：Mulbackward0:乘法运算节点，反向传播时用来算梯度。
# 一句话：`MulBackward0` 就是一个**小账本**，记录：这个结果来自乘法，并且存好了乘法对应的求导公式，等后面调用 `.backward()` 的时候拿出来算导数。
# 当你写 `y.backward()`，PyTorch 就找到这个`MulBackward0`，调用里面的求导公式算出梯度。



# 调用 `y.backward()`反向求导：**沿着计算图反向，算所有 requires_grad=True 张量的梯度，结果存在 .grad 属性**
y.backward()
# w的梯度，存在w.grad
print("w的梯度 dw：", w.grad) # 输出 tensor(3.)

# 1. 只有标量（单个数字）能直接.backward ()。如果 y 是向量，需要传`y.sum().backward()`
# 2. **每次 backward，梯度会累加！不会自动清零**，训练循环里必须 `w.grad.zero_()` 清零，不然梯度爆炸。这个是新手最大坑！

### 梯度下降公式回顾



#五：梯度下降更新参数，拟合曲线，实战代码操作

# 目标：拟合一条直线 y = w * x
# 真实关系：假设真实 w=5，也就是 y=5x
# 我们给一堆 x，带一点噪声的 y 数据；
# 我们初始化猜测 w，不断用梯度下降更新 w，让 w 慢慢逼近 5。


import torch

# 1. 准备手里真实的数据集
x = torch.tensor([1.,2.,3.,4.,5.])
y_real = torch.tensor([5.,10.,15.,20.,25.]) # 真实 w=5

# 2. 初始化待训练参数w，先猜一个初始的w，开启求导
w = torch.tensor(1.0, requires_grad=True)     # requires_grad=True记录他每时每刻的变化生成计算图
lr = 0.01 # 学习率
epoch = 20 # 迭代轮数

for i in range(epoch):            #迭代循环20次
    # 正向传播：算预测值，算loss
    y_pred = w * x       #这里的w是猜测的
    loss = torch.mean((y_pred - y_real)**2) # MSE均方误差

    #不手动微分求梯度        求的是loss对w的梯度
    # 利用反向传播：自动求梯度  求出梯度我就知道，w往哪里走，可以让loss降低
    loss.backward()    #backward求的就是开了rquires_grad的梯度    结果存在.grad里面

    # 更新参数！这一步必须放在 torch.no_grad() 里面
    # 因为更新w属于手动赋值，不要让autograd记录这次操作
    with torch.no_grad():
        w -= lr * w.grad   # w.grad  就是loss对w的微分  w的梯度

    # 梯度清零！！！极其重要，不清零梯度会叠加
    w.grad.zero_()

    if i%2 ==0:
        print(f"轮数{i:2d} | w={w.item():.3f}, loss={loss.item():.3f}")


# # 放到代码里的流程
# 1. 拿当前参数 w，计算预测值y_pred=w*x
# 2. 算损失 loss：预测值和真实值差多少
# 3. `loss.backward()`：Autograd 自动求梯度，知道 w 往哪边改能减小 loss
# 4. w = w - lr.w.grad：更新 w（下山一步）    w.grad为梯度
# 5. `w.grad.zero_()`：清空梯度，准备下一轮
# 6. 循环很多次，w 慢慢收敛到合适的值