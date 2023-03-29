import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.sinh(x)
print(out)
# [-0.41075233 -0.201336    0.10016675  0.30452029]