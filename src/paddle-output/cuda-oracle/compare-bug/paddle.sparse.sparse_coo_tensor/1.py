results = dict()
import paddle
arg_1_0_0 = 0
arg_1_0_1 = 1
arg_1_0_2 = 2
arg_1_0 = [arg_1_0_0,arg_1_0_1,arg_1_0_2,]
arg_1_1_0 = 1
arg_1_1_1 = 2
arg_1_1_2 = 0
arg_1_1 = [arg_1_1_0,arg_1_1_1,arg_1_1_2,]
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = 1.0
arg_2_1 = 2.0
arg_2_2 = 3.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = 3
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = [arg_1_0_0,arg_1_0_1,arg_1_0_2,]
arg_1_1 = [arg_1_1_0,arg_1_1_1,arg_1_1_2,]
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
