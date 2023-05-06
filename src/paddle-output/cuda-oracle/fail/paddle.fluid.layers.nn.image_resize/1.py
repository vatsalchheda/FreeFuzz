results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048,4,[2, 3, 6, 51], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 12
arg_2_1 = 12
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = None
arg_4 = None
arg_5 = "BILINEAR"
arg_6 = None
arg_7 = True
arg_8 = 1
arg_9 = "NCHW"
try:
  results["res_cpu"] = paddle.fluid.layers.nn.image_resize(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.image_resize(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
