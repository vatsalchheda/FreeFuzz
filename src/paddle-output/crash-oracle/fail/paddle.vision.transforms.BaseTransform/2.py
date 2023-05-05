import paddle
arg_1 = None
arg_class = paddle.vision.transforms.BaseTransform(arg_1,)
int_tensor = paddle.randint(low=0, high=256, shape=[1, 28, 28], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_0_tensor = uint8_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
