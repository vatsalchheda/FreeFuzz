import paddle
arg_1 = 1
arg_class = paddle.nn.BatchNorm3D(arg_1,)
int_tensor = paddle.randint(low=-32768, high=32767, shape=[16, 1, 2, 2, 3], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_2_0_tensor = int16_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
