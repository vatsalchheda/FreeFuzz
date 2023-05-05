results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,1,[2, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1027.0
arg_3_1 = "mean"
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,shape=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,shape=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
