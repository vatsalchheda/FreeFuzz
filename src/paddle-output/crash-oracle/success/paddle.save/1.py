import paddle
arg_1 = "paddlenlp.trainer.training_argsTrainingArguments"
arg_2 = "training_checkpoints/checkpoint-3/training_args.bin"
res = paddle.save(arg_1,arg_2,)
