results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 454
arg_3 = 21
arg_4 = "tanh"
arg_5 = 66.0
try:
  results["res_cpu"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
