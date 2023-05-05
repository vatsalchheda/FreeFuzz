results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 32768, [4], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-64, 2, [4], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
try:
  results["res_cpu"] = paddle.incubate.graph_send_recv(arg_1,arg_2,arg_3,pool_type=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.incubate.graph_send_recv(arg_1,arg_2,arg_3,pool_type=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
