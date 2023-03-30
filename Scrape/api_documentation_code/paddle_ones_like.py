import paddle

x = paddle.to_tensor([1,2,3])
out1 = paddle.ones_like(x) # [1., 1., 1.]
out2 = paddle.ones_like(x, dtype='int32') # [1, 1, 1]