results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,16,[100, 1, 28, 28], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
arg_3 = 20
arg_4 = 2
arg_5 = 2
arg_6 = "relu"
try:
  results["res_cpu"] = paddle.fluid.nets.simple_img_conv_pool(input=arg_1,filter_size=arg_2,num_filters=arg_3,pool_size=arg_4,pool_stride=arg_5,act=arg_6,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.nets.simple_img_conv_pool(input=arg_1,filter_size=arg_2,num_filters=arg_3,pool_size=arg_4,pool_stride=arg_5,act=arg_6,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
