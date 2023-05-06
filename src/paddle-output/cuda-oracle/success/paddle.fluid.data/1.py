results = dict()
import paddle
arg_1 = "roi_2"
arg_2_0 = None
arg_2_1 = -48
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "replicate"
arg_4 = -21
try:
  results["res_cpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
