import paddle

x = [[2,3,4], [7,8,9]]
x = paddle.to_tensor(x, dtype='float32')
res = paddle.log(x)
# [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]