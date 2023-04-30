results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,1,[2, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -115.0
arg_3 = 1
arg_4 = 0
try:
  results["res_cpu"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
