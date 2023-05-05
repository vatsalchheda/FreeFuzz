results = dict()
import paddle
arg_1_tensor = paddle.rand([4, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -55
arg_2_1 = 24
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "bool"
arg_4 = 1e+20
arg_5 = True
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(input=arg_1,shape=arg_2,dtype=arg_3,value=arg_4,force_cpu=arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(input=arg_1,shape=arg_2,dtype=arg_3,value=arg_4,force_cpu=arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
