import paddle
arg_1 = "./infer_model.params"
arg_2 = True
res = paddle.static.save_to_file(arg_1,arg_2,)
