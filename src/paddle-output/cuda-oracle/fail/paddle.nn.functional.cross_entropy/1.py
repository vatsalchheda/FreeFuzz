results = dict()
import paddle
arg_1_tensor = paddle.rand([0, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,4096,[32, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = -16
arg_5 = "mean"
arg_6 = False
arg_7 = -1
arg_8 = True
arg_9 = None
try:
  results["res_cpu"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
