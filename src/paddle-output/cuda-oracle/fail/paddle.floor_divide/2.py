results = dict()
import paddle
int_tensor = paddle.randint(low=0, high=256, shape=[17], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_1_tensor = uint8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384, 1, [], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.floor_divide(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.floor_divide(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
