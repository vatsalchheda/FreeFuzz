results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,1,[1, 6, 1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = "NCHW"
arg_4 = None
try:
  results["res_cpu"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.channel_shuffle(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
