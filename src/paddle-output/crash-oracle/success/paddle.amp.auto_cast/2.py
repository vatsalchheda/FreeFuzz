import paddle
arg_1 = True
arg_2 = 3
arg_3 = True
res = paddle.amp.auto_cast(enable=arg_1,custom_white_list=arg_2,level=arg_3,)
