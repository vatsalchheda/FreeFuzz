results = dict()
import paddle
import time
arg_1 = "paddlenlp.transformers.ernie.modelingErnieForSequenceClassification"
arg_2 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\finetune_static\model"
start = time.time()
results["time_low"] = paddle.jit.save(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.save(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
