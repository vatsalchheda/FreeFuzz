results = dict()
import paddle
arg_1 = 2
arg_2 = 1
arg_3 = 12
arg_class = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
int_tensor = paddle.randint(low=-32768, high=32767, shape=[47, 3, 32], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_4_0_tensor = int16_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
