results = dict()
import paddle
arg_1 = "cond"
arg_2 = "body"
arg_3_0_tensor = paddle.randint(-4096, 32768, [1], dtype=paddle.int64arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
