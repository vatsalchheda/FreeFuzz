results = dict()
import paddle
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "MM"
arg_3 = "github"
try:
  results["res_cpu"] = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
