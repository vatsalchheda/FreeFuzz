import paddle
arg_1 = 16
arg_2 = 32
arg_class = paddle.nn.LSTMCell(arg_1,arg_2,)
int_tensor = paddle.randint(low=0, high=255, shape=[4, 72], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_3_0_tensor = uint8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 0], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
