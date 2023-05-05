import paddle
int_tensor = paddle.randint(low=-32768, high=32768, shape=[11, 4], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_1_tensor = int16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16, 512, [11, 4, 4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.gather_tree(arg_1,arg_2,)
