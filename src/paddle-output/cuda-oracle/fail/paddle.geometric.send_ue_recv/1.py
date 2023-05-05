results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-256,4096,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.randint(-64,1024,[3], dtype=paddle.int32)
arg_4 = arg_4_tensor.clone()
arg_5 = "add"
arg_6 = "sum"
arg_7_tensor = paddle.randint(-16,16,[1], dtype=paddle.int32)
arg_7 = arg_7_tensor.clone()
try:
  results["res_cpu"] = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,out_size=arg_7,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
arg_7 = arg_7_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.geometric.send_ue_recv(arg_1,arg_2,arg_3,arg_4,message_op=arg_5,reduce_op=arg_6,out_size=arg_7,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
