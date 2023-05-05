import paddle
int_tensor = paddle.randint(low=-32768, high=32767, shape=[1], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_1_tensor = int16_tensor
arg_1 = arg_1_tensor.clone()
res = paddle.mean(arg_1,)
