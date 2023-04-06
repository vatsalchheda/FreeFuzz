import paddle

input = paddle.rand((3, 100, 100))
rank = paddle.rank(input)
print(rank)
# 3