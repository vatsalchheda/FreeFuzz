results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,4096,[512, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0.1
arg_2_1 = 0.1
arg_2_2 = 0.2
arg_2_3 = 0.2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3_tensor = paddle.randint(-8192,32768,[512, 81, 4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "decode_center_size"
arg_5 = False
arg_6 = 1
arg_7 = None
try:
  results["res_cpu"] = paddle.vision.ops.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,box_normalized=arg_5,axis=arg_6,name=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.vision.ops.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,box_normalized=arg_5,axis=arg_6,name=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
