import paddle

x = paddle.to_tensor([1, 2, 3])
out1 = paddle.zeros_like(x) # [0., 0., 0.]
out2 = paddle.zeros_like(x, dtype='int32') # [0, 0, 0]