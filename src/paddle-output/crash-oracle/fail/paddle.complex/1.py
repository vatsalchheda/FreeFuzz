import paddle
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 7, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.complex(arg_1,arg_2,)
