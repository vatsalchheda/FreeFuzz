results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\.paddlespeech\models\conformer_wenetspeech-zh-16k\1.0\asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar\exp\conformer\checkpoints\wenetspeech.pdparams"
start = time.time()
results["time_low"] = paddle.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
