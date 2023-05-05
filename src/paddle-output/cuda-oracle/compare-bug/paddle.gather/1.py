results = dict()
import paddle
float_tensor = paddle.rand([28, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 2, [10], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.gather(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.gather(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
