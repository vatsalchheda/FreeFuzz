results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048, 8192, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-1, 128, [7], dtype=paddle.int64arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-4, 4, [5], dtype=paddle.int64arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_tensor = paddle.randint(-4, 16384, [3], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2_1 = arg_2_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
