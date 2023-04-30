results = dict()
import paddle
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = -1.0
arg_3 = False
try:
  results["res_cpu"] = paddle.hub.list(arg_1,source=arg_2,force_reload=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.hub.list(arg_1,source=arg_2,force_reload=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
