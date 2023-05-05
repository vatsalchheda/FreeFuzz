results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-8192,1024,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = False
try:
  results["res_cpu"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
