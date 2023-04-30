results = dict()
import paddle
arg_1_tensor = paddle.randint(-256,16384,[2, 1, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
try:
  results["res_cpu"] = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.pixel_unshuffle(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
