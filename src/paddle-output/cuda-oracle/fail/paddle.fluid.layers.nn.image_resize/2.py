results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 3, 6, 9], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "BILINEAR"
try:
  results["res_cpu"] = paddle.fluid.layers.nn.image_resize(input=arg_1,out_shape=arg_2,resample=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.image_resize(input=arg_1,out_shape=arg_2,resample=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
