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

    if i%2 ==0:   #i为偶数时才打印 w和 loss
        print(f"轮数{i:2d} | w={w.item():.3f}, loss={loss.item():.3f}")  


# ：2d - `:`：格式化的分隔符
# - `2`：**最小占 2 个字符宽度**
# - `d`：代表输出**整数（decimal）**    .3f ;保留三位有效数字
