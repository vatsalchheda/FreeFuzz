results = dict()
import paddle
arg_1 = -0.35355339059327373
arg_2 = 0.35355339059327373
arg_3 = 0
arg_4 = -39
arg_5 = 0
arg_6 = -1.0
arg_class = paddle.fluid.initializer.UniformInitializer(low=arg_1,high=arg_2,seed=arg_3,diag_num=arg_4,diag_step=arg_5,diag_val=arg_6,)
arg_7_0_tensor = paddle.rand([32], dtype=paddle.float32)
arg_7_0 = arg_7_0_tensor.clone()
arg_7_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_7_1 = arg_7_1_tensor.clone()
arg_7 = [arg_7_0,arg_7_1,]
try:
  results["res_cpu"] = arg_class(*arg_7)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_7_0 = arg_7_0_tensor.clone().cuda()
arg_7_1 = arg_7_1_tensor.clone().cuda()
arg_7 = [arg_7_0,arg_7_1,]
try:
  results["res_gpu"] = arg_class(*arg_7)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
