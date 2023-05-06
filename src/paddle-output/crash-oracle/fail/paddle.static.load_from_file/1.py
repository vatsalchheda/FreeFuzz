import paddle
arg_1 = "./infer_model.params"
res = paddle.static.load_from_file(arg_1,)
