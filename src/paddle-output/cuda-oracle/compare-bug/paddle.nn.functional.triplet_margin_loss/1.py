results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = 2
arg_6 = 63.0
arg_7 = False
arg_8 = "none"
arg_9 = None
try:
  results["res_cpu"] = paddle.nn.functional.triplet_margin_loss(arg_1,arg_2,arg_3,margin=arg_4,p=arg_5,epsilon=arg_6,swap=arg_7,reduction=arg_8,name=arg_9,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.triplet_margin_loss(arg_1,arg_2,arg_3,margin=arg_4,p=arg_5,epsilon=arg_6,swap=arg_7,reduction=arg_8,name=arg_9,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
