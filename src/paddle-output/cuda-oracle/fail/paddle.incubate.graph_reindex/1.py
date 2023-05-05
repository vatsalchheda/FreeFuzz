results = dict()
import paddle
arg_1_tensor = paddle.randint(-512, 16, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 256, [12], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1, 16384, [6], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.incubate.graph_reindex(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.incubate.graph_reindex(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
