results = dict()
import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
float_tensor = paddle.rand([1, 3, 0], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_tensor = f16_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = "reflect"
arg_4 = -15.0
arg_5 = -24.0
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
