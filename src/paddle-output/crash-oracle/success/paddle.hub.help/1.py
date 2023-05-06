import paddle
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "MM"
arg_3 = "github"
res = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
