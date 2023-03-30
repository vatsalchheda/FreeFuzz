import paddle

x = paddle.to_tensor([1, 4, 5, 2])
out = paddle.diff(x)
print(out)
# out:
# [3, 1, -3]

y = paddle.to_tensor([7, 9])
out = paddle.diff(x, append=y)
print(out)
# out:
# [3, 1, -3, 5, 2]

z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
out = paddle.diff(z, axis=0)
print(out)
# out:
# [[3, 3, 3]]
out = paddle.diff(z, axis=1)
print(out)
# out:
# [[1, 1], [1, 1]]