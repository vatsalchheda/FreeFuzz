import paddle

x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
out = paddle.sqrt(x)
print(out)
# [0.31622777 0.4472136  0.54772256 0.63245553]