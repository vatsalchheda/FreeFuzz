results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 4, 20, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = False
arg_4 = "wrap"
try:
  results["res_cpu"] = paddle.nn.functional.dropout(arg_1,arg_2,training=arg_3,mode=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.dropout(arg_1,arg_2,training=arg_3,mode=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
