results = dict()
import paddle
arg_1 = 128
arg_2 = True
arg_class = paddle.incubate.nn.FusedMultiHeadAttention(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3_2 = arg_3_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
