results = dict()
import paddle
arg_1 = -47
arg_2 = 2
arg_3 = 0
arg_4 = True
arg_class = paddle.nn.MaxPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,return_mask=arg_4,)
arg_5_0_tensor = paddle.randint(-128,4,[1, 3, 32, 32], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
try:
  results["res_cpu"] = arg_class(*arg_5)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_5_0 = arg_5_0_tensor.clone().cuda()
arg_5 = [arg_5_0,]
try:
  results["res_gpu"] = arg_class(*arg_5)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
