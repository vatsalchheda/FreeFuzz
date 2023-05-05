results = dict()
import paddle
int_tensor = paddle.randint(low=0, high=256, shape=[61, 4, 1], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_1_tensor = uint8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 37
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
