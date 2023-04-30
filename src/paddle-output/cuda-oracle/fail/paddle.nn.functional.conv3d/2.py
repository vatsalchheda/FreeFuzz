results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048,2,[2, 3, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2,[0, 3, 3, 0, 52], dtype=paddle.bfloat16)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
