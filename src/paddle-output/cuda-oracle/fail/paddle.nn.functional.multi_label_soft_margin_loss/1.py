results = dict()
import paddle
arg_1_tensor = paddle.randint(-2,2048,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,512,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "none"
try:
  results["res_cpu"] = paddle.nn.functional.multi_label_soft_margin_loss(arg_1,arg_2,reduction=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.multi_label_soft_margin_loss(arg_1,arg_2,reduction=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
