results = dict()
import paddle
arg_1 = 3
arg_2 = True
arg_class = paddle.nn.AdaptiveMaxPool3D(output_size=arg_1,return_mask=arg_2,)
arg_3_0_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
