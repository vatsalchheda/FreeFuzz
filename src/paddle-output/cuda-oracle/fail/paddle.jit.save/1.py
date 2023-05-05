results = dict()
import paddle
arg_1 = "matching"
arg_2 = "__internal_testing__/tiny-random-rocketqa-cross-encoder\static\inference"
try:
  results["res_cpu"] = paddle.jit.save(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.save(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
