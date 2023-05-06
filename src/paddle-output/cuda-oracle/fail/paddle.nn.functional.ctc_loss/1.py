results = dict()
import paddle
arg_1_tensor = paddle.rand([5, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 16384, [2, 3], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16, 256, [2], dtype=paddle.int64arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-8, 16384, [2], dtype=paddle.int64arg_4 = arg_4_tensor.clone()
arg_5 = -1024
arg_6 = "none"
arg_7 = False
try:
  results["res_cpu"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.ctc_loss(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,norm_by_times=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
