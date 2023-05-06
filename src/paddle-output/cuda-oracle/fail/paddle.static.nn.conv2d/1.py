results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 1, 28, 28], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 6
arg_3 = 3
try:
  results["res_cpu"] = paddle.static.nn.conv2d(input=arg_1,num_filters=arg_2,filter_size=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.nn.conv2d(input=arg_1,num_filters=arg_2,filter_size=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
