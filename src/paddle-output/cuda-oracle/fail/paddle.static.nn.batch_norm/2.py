results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 6, 26, 26], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "reflect"
try:
  results["res_cpu"] = paddle.static.nn.batch_norm(input=arg_1,act=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.nn.batch_norm(input=arg_1,act=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
