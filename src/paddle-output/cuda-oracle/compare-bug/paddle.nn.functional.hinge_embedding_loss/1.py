results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "none"
arg_4 = 1.0
arg_5 = None
try:
  results["res_cpu"] = paddle.nn.functional.hinge_embedding_loss(arg_1,arg_2,reduction=arg_3,margin=arg_4,name=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.hinge_embedding_loss(arg_1,arg_2,reduction=arg_3,margin=arg_4,name=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
