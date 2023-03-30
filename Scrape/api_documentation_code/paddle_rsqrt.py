import paddle

x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
out = paddle.rsqrt(x)
print(out)
# [3.16227766 2.23606798 1.82574186 1.58113883]