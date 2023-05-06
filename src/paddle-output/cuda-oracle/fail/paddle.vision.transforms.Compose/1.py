results = dict()
import paddle
arg_1 = []
arg_class = paddle.vision.transforms.Compose(arg_1,)
int_tensor = paddle.randint(low=0, high=255, shape=[0, 2], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_0_tensor = uint8_tensor
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
