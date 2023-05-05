import paddle
int_tensor = paddle.randint(low=-32768, high=32767, shape=[2, 0, 8, 8, 8], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_1_tensor = int16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_tensor = paddle.rand([32], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0
arg_5 = 16
arg_6_0 = -7
arg_6_1 = 3
arg_6_2 = -23
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7_0 = -32
arg_7_1 = -38
arg_7_2 = 0
arg_7 = [arg_7_0,arg_7_1,arg_7_2,]
arg_8 = 1
arg_9 = None
arg_10 = "NCDHW"
res = paddle.nn.functional.conv3d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
