import paddle
x = [[0, 1, 2],
[3, 4, 5]]

y = [[6, 7 ,8],
[9, 10, 11]]

output = paddle.fluid.contrib.layers.nn.partial_sum([x, y], start_index=0, length=2)