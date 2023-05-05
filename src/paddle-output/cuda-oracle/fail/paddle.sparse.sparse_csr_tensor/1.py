results = dict()
import paddle
arg_1_0 = 43
arg_1_1 = -16
arg_1_2 = 57
arg_1_3 = 15
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2_0 = 1
arg_2_1 = 3
arg_2_2 = 2
arg_2_3 = 0
arg_2_4 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
arg_3_0 = 1
arg_3_1 = 2
arg_3_2 = 3
arg_3_3 = 4
arg_3_4 = 5
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4_0 = 3
arg_4_1 = 3
arg_4 = [arg_4_0,arg_4_1,]
try:
  results["res_cpu"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4 = [arg_4_0,arg_4_1,]
try:
  results["res_gpu"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
