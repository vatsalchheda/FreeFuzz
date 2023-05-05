results = dict()
import paddle
arg_1_0 = 20
arg_1_1 = 32
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "emb.w"
arg_3 = False
arg_class = paddle.fluid.dygraph.nn.Embedding(size=arg_1,param_attr=arg_2,is_sparse=arg_3,)
arg_4_0_tensor = paddle.randint(-4096,256,[1], dtype=paddle.int64)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
