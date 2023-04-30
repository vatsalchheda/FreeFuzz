results = dict()
import paddle
arg_1 = "paddlenlp.trainer.training_argsTrainingArguments"
arg_2 = "training_checkpoints/checkpoint-3/training_args.bin"
try:
  results["res_cpu"] = paddle.save(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.save(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
