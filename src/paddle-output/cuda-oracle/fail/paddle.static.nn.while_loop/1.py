results = dict()
import paddle
arg_1 = "cond_zoom"
arg_2 = "body_zoom"
arg_3_0_tensor = paddle.randint(-1,1024,[1], dtype=paddle.int64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(0,2,[1], dtype=paddle.bool)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-256,32,[1], dtype=paddle.float32)
arg_3_2 = arg_3_2_tensor.clone()
arg_3_3_tensor = paddle.randint(-1,8192,[1], dtype=paddle.float32)
arg_3_3 = arg_3_3_tensor.clone()
arg_3_4_tensor = paddle.randint(-4096,512,[1], dtype=paddle.float32)
arg_3_4 = arg_3_4_tensor.clone()
arg_3_5_tensor = paddle.randint(-4,512,[2], dtype=paddle.float32)
arg_3_5 = arg_3_5_tensor.clone()
arg_3_6_tensor = paddle.randint(-16,4096,[1], dtype=paddle.float32)
arg_3_6 = arg_3_6_tensor.clone()
arg_3_7_tensor = paddle.randint(-32768,1024,[1], dtype=paddle.float32)
arg_3_7 = arg_3_7_tensor.clone()
arg_3_8_tensor = paddle.randint(-2,1024,[1], dtype=paddle.float32)
arg_3_8 = arg_3_8_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
try:
  results["res_cpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3_2 = arg_3_2_tensor.clone().cuda()
arg_3_3 = arg_3_3_tensor.clone().cuda()
arg_3_4 = arg_3_4_tensor.clone().cuda()
arg_3_5 = arg_3_5_tensor.clone().cuda()
arg_3_6 = arg_3_6_tensor.clone().cuda()
arg_3_7 = arg_3_7_tensor.clone().cuda()
arg_3_8 = arg_3_8_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,arg_3_7,arg_3_8,]
try:
  results["res_gpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
