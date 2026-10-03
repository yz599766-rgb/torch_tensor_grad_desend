# 代码检验梯度下降
import torch

x=torch.tensor([1.,2.,3.,4.,5.,6.])
y_real=torch.tensor([5.,10.,15.,20.,25.,30.])


w=torch.tensor(2.0, requires_grad=True)
lr=0.01
epoch=20

for i in range(epoch):
    y_pred=w * x
    loss=torch.mean((y_pred-y_real)**2)   #torch.mean标准MSE写法


    loss.backward()

    with torch.no_grad():
           w-=lr*w.grad


    w.grad.zero_()



    if i%2==0:
         print(f"轮数:{i:2d}  |  w={w.item():.3f}  loss={loss.item():.3f}")


#注意点：for以下内容全部都要在for循环里面才可以








