results = dict()
import paddle
arg_1_tensor = paddle.randint(-32, 512, [1, 26], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([513, 768], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = True
arg_5 = None
try:
  results["res_cpu"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
