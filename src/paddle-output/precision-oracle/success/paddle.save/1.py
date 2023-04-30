results = dict()
import paddle
import time
arg_1 = "paddlenlp.trainer.training_argsTrainingArguments"
arg_2 = "training_checkpoints/checkpoint-3/training_args.bin"
start = time.time()
results["time_low"] = paddle.save(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.save(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
