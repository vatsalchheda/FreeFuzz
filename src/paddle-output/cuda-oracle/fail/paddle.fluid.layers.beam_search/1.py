results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,8192,[-1, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-64,4096,[-1, 4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([0, 4], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5 = "max"
arg_6 = 1
try:
  results["res_cpu"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
