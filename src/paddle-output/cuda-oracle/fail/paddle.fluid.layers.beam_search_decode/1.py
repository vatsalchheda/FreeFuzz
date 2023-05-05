results = dict()
import paddle
arg_1_tensor = paddle.randint(-1,1,[], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 960
arg_4 = 42
try:
  results["res_cpu"] = paddle.fluid.layers.beam_search_decode(arg_1,arg_2,beam_size=arg_3,end_id=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.beam_search_decode(arg_1,arg_2,beam_size=arg_3,end_id=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
