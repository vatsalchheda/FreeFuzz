results = dict()
import paddle
arg_1 = 8
arg_2 = 125
arg_3 = 40.01
try:
  results["res_cpu"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
