results = dict()
import paddle
arg_1_tensor = paddle.randint(-256,1,[2, 2, 1, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = True
arg_4 = "NCHW"
arg_5 = None
try:
  results["res_cpu"] = paddle.nn.functional.dropout2d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.dropout2d(arg_1,p=arg_2,training=arg_3,data_format=arg_4,name=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
