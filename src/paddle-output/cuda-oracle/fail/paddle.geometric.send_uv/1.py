results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192,4096,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,512,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,1,[4], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-32,8,[4], dtype=paddle.int32)
arg_4 = arg_4_tensor.clone()
arg_5 = "add"
try:
  results["res_cpu"] = paddle.geometric.send_uv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.geometric.send_uv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
