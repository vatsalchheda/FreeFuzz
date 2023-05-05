results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,512,[11, 4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,16384,[11, 4, 4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.nn.gather_tree(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.gather_tree(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
