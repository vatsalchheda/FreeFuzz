results = dict()
import paddle
arg_1 = "C:\Users\phalt\.paddlespeech\models\conformer_wenetspeech-zh-16k\1.0\asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar\exp\conformer\checkpoints\wenetspeech.pdparams"
try:
  results["res_cpu"] = paddle.load(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.load(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
