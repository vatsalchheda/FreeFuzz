results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 4, 0, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0_tensor = paddle.randint(-2, 4, [1], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0 = -60
arg_4_1 = 25
arg_4_2 = 63
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = 19
arg_5_1 = -1024
arg_5_2 = -16
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
try:
  results["res_cpu"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
try:
  results["res_gpu"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
