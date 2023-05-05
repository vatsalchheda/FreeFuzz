results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048, 32, [64, 0], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8, 32, [64, 32, 32], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 22.0
try:
  results["res_cpu"] = paddle.fluid.layers.nn.mean_iou(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.mean_iou(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
