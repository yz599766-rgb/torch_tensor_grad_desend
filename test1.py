import torch

# 1. 初始化参数：随便猜一个w，初始是2
w = torch.tensor(2.0, requires_grad=True)
lr = 0.01 # 学习率：每一步迈多小

# 真实数据：y_real = 5*x
x = torch.tensor(4.0)
y_real = torch.tensor(20.0)

print("====开始训练，目标w=5====")
for epoch in range(200):
    # 正向传播：用当前w做预测
    y_pred = w * x
    loss = (y_pred - y_real) ** 2 # MSE损失：预测值和真实值差的平方
    
    # 自动求导，Autograd干活
    loss.backward()
    
    # 更新参数：w = w - lr * 梯度
    # with torch.no_grad()：参数更新这一步不要记录计算图！
    with torch.no_grad():
        w -= lr * w.grad
    
    # ✅梯度清零！！不写这里梯度会累加，直接爆炸
    w.grad.zero_()

    # 每20轮打印一次，观察w和loss变化
    if epoch % 20 == 0:
        print(f"轮次epoch={epoch:3d} | w={w.item():.4f} | loss={loss.item():.4f}")
