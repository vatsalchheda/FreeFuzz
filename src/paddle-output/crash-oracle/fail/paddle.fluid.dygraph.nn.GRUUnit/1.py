import paddle
arg_1 = 1
arg_class = paddle.fluid.dygraph.nn.GRUUnit(size=arg_1,)
int_tensor = paddle.randint(low=0, high=255, shape=[9, 15], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_0_tensor = uint8_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([9, 0, 1], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = arg_class(*arg_2)
