results = dict()
import paddle
arg_1 = "paddlenlp.transformers.ernie.modelingErnieForSequenceClassification"
arg_2 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\finetune_static\model"
try:
  results["res_cpu"] = paddle.jit.save(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.save(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
