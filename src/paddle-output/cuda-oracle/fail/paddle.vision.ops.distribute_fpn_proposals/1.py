results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 98
arg_3 = 5
arg_4 = 4
arg_5 = 224
arg_6 = None
arg_7 = None
try:
  results["res_cpu"] = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.vision.ops.distribute_fpn_proposals(fpn_rois=arg_1,min_level=arg_2,max_level=arg_3,refer_level=arg_4,refer_scale=arg_5,rois_num=arg_6,name=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
