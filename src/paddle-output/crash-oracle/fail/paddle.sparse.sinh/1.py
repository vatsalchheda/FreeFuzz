import paddle
int_tensor = paddle.randint(low=-128, high=127, shape=[], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
res = paddle.sparse.sinh(arg_1,)
