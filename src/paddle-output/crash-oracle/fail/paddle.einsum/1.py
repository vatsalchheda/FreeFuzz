import paddle
arg_1 = "ijk->kji"
int_tensor = paddle.randint(low=0, high=255, shape=[2, 16, 2], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_tensor = uint8_tensor
arg_2 = arg_2_tensor.clone()
res = paddle.einsum(arg_1,arg_2,)
