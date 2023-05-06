results = dict()
import paddle
arg_1_tensor = paddle.rand([192], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([192], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4 = 47.00000001
try:
  results["res_cpu"] = paddle.nn.functional.cosine_similarity(arg_1,arg_2,axis=arg_3,eps=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.cosine_similarity(arg_1,arg_2,axis=arg_3,eps=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
