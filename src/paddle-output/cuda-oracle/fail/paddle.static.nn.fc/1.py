results = dict()
import paddle
arg_1_tensor = paddle.randint(-16,512,[-1, 784], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 10
arg_3 = "softmax"
try:
  results["res_cpu"] = paddle.static.nn.fc(arg_1,size=arg_2,activation=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.nn.fc(arg_1,size=arg_2,activation=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
