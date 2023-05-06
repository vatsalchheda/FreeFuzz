import paddle
int_tensor = paddle.randint(low=0, high=256, shape=[61, 4, 1], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_1_tensor = uint8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 37
arg_2 = [arg_2_0,]
res = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
