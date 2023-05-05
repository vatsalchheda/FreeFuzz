results = dict()
import paddle
arg_1 = True
arg_2 = 3
arg_3 = True
try:
  results["res_cpu"] = paddle.amp.auto_cast(enable=arg_1,custom_white_list=arg_2,level=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.amp.auto_cast(enable=arg_1,custom_white_list=arg_2,level=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
