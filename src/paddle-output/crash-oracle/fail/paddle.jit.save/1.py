import paddle
arg_1 = "paddlenlp.transformers.ernie.modelingErnieForSequenceClassification"
arg_2 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\finetune_static\model"
res = paddle.jit.save(arg_1,arg_2,)
