results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3, 8, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = True
arg_4 = None
try:
  results["res_cpu"] = paddle.nn.functional.adaptive_max_pool3d(arg_1,output_size=arg_2,return_mask=arg_3,name=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.adaptive_max_pool3d(arg_1,output_size=arg_2,return_mask=arg_3,name=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
