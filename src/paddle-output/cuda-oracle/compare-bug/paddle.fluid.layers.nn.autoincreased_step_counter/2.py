results = dict()
import paddle
arg_1 = "@LR_DECAY_COUNTER@"
arg_2 = 0
arg_3 = -15
try:
  results["res_cpu"] = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
