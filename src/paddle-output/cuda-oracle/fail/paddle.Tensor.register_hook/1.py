results = dict()
import paddle
int_tensor = paddle.randint(low=-32768, high=32767, shape=[4], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_1_tensor = int16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = "double_hook_fn"
try:
  results["res_cpu"] = paddle.Tensor.register_hook(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.Tensor.register_hook(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
