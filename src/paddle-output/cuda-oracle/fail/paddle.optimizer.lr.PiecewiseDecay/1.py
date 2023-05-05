results = dict()
import paddle
arg_1_0 = 5
arg_1_1 = 8
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = False
arg_2_1 = "max"
arg_2_2 = True
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_class = paddle.optimizer.lr.PiecewiseDecay(boundaries=arg_1,values=arg_2,)
arg_3 = []
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3 = []
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
