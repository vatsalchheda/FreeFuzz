results = dict()
import paddle
arg_1_tensor = paddle.randint(-16,8192,[6, 4, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 0.2
try:
  results["res_cpu"] = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
