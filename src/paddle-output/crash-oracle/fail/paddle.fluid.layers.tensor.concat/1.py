import paddle
int_tensor = paddle.randint(low=0, high=255, shape=[21, 23, 32, 1], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_1_0_tensor = uint8_tensor
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-16,1024,[2, 86, 32], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = -26
res = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
