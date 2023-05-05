import paddle
arg_1_tensor = paddle.rand([-1, 20, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
arg_3 = 20
arg_4 = "reflect"
res = paddle.static.nn.conv2d(input=arg_1,filter_size=arg_2,num_filters=arg_3,act=arg_4,)
