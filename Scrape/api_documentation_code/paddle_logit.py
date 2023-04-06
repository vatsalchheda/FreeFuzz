import paddle

x = paddle.to_tensor([0.2635, 0.0106, 0.2780, 0.2097, 0.8095])
out1 = paddle.logit(x)
print(out1)
# [-1.0277, -4.5365, -0.9544, -1.3269,  1.4468]