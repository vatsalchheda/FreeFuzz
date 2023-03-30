import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.expm1(x)
print(out)
# [-0.32967997, -0.18126924,  0.10517092,  0.34985882]