import paddle
arg_1 = -5
arg_2 = 0.17677669529663687
arg_class = paddle.nn.initializer.Uniform(arg_1,arg_2,)
int_tensor = paddle.randint(low=0, high=255, shape=[128, 32], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_3_0_tensor = uint8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
