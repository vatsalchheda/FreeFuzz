import paddle

data = paddle.arange(4)
data = paddle.reshape(data, (2, 2))
print(data)
#[[0, 1],
# [2, 3]]

y = paddle.rot90(data, 1, [0, 1])
print(y)
#[[1, 3],
# [0, 2]]

y= paddle.rot90(data, -1, [0, 1])
print(y)
#[[2, 0],
# [3, 1]]

data2 = paddle.arange(8)
data2 = paddle.reshape(data2, (2,2,2))
print(data2)
#[[[0, 1],
#  [2, 3]],
# [[4, 5],
#  [6, 7]]]

y = paddle.rot90(data2, 1, [1, 2])
print(y)
#[[[1, 3],
#  [0, 2]],
# [[5, 7],
#  [4, 6]]]