import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.atan(x)
print(out)
# [-0.38050638 -0.19739556  0.09966865  0.29145679]