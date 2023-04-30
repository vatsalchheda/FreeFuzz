import paddle
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = -1.0
arg_3 = False
res = paddle.hub.list(arg_1,source=arg_2,force_reload=arg_3,)
