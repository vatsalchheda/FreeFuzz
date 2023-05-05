results = dict()
import paddle
arg_1_tensor = paddle.rand([10, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([10, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([2, 21, 4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "decode_center_size"
try:
  results["res_cpu"] = paddle.fluid.layers.detection.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.detection.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
