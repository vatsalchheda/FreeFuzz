import paddle

out1 = paddle.randperm(5)
# [4, 1, 2, 3, 0]  # random

out2 = paddle.randperm(7, 'int32')
# [1, 6, 2, 0, 4, 3, 5]  # random