results = dict()
import paddle
arg_1_tensor = paddle.randint(-128, 512, [4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 12
arg_3 = "paddleVarType"
try:
  results["res_cpu"] = paddle.fluid.layers.sequence_lod.sequence_mask(arg_1,maxlen=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.sequence_lod.sequence_mask(arg_1,maxlen=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
