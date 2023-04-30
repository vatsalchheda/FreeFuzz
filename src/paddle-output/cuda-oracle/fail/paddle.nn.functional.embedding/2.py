results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,4,[3, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,4,[10, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4 = "embedding"
try:
  results["res_cpu"] = paddle.nn.functional.embedding(x=arg_1,weight=arg_2,sparse=arg_3,name=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.embedding(x=arg_1,weight=arg_2,sparse=arg_3,name=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
