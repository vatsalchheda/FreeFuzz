results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 256, [3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4096, 4, [3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4 = "sum"
arg_5_tensor = paddle.randint(-4096, 8, [1], dtype=paddle.int32)
arg_5 = arg_5_tensor.clone()
try:
  results["res_cpu"] = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,out_size=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_5 = arg_5_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.geometric.send_u_recv(arg_1,arg_2,arg_3,reduce_op=arg_4,out_size=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
