results = dict()
import paddle
arg_1 = 4
arg_2 = -49
arg_class = paddle.nn.SimpleRNNCell(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-8,16384,[0, 43], dtype=paddle.bfloat16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([4, 28], dtype=paddle.float64)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
