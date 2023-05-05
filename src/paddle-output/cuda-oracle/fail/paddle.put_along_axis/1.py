results = dict()
import paddle
arg_1_tensor = paddle.randint(-32,256,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([57, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = -1008.0
arg_4 = 0
try:
  results["res_cpu"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
