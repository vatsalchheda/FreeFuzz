results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 1, 28, 28], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -37
arg_3_0 = 3
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = False
arg_5 = "relu"
arg_6 = -8
arg_7 = 2
try:
  results["res_cpu"] = paddle.fluid.nets.img_conv_group(input=arg_1,conv_padding=arg_2,conv_num_filter=arg_3,conv_filter_size=arg_4,conv_act=arg_5,pool_size=arg_6,pool_stride=arg_7,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fluid.nets.img_conv_group(input=arg_1,conv_padding=arg_2,conv_num_filter=arg_3,conv_filter_size=arg_4,conv_act=arg_5,pool_size=arg_6,pool_stride=arg_7,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
