results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-8192,4096,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = "sum"
arg_6 = None
try:
  results["res_cpu"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
