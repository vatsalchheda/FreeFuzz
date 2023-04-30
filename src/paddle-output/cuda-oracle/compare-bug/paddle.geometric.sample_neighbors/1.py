results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384,4,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,4,[11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1,2048,[4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = 2
try:
  results["res_cpu"] = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
