results = dict()
import paddle
arg_1 = 2
arg_2 = 8
arg_class = paddle.nn.Embedding(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-32768, 2048, [1, 1], dtype=paddle.int64arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
