results = dict()
import paddle
arg_1_tensor = paddle.randint(-1,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-8,128,[7], dtype=paddle.int64)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-8192,8192,[5], dtype=paddle.int64)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0_tensor = paddle.randint(-8192,256,[3], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-64,1,[3], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2_1 = arg_2_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
