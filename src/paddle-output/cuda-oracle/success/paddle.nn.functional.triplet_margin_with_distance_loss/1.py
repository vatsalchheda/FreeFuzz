results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
int_tensor = paddle.randint(low=-32768, high=32767, shape=[3, 3], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_3_tensor = int16_tensor
arg_3 = arg_3_tensor.clone()
arg_4 = 1.0
arg_5 = False
arg_6 = "mean"
arg_7 = None
try:
  results["res_cpu"] = paddle.nn.functional.triplet_margin_with_distance_loss(arg_1,arg_2,arg_3,margin=arg_4,swap=arg_5,reduction=arg_6,name=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.triplet_margin_with_distance_loss(arg_1,arg_2,arg_3,margin=arg_4,swap=arg_5,reduction=arg_6,name=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
