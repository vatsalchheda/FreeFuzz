import paddle
arg_1 = "matching"
arg_2 = "__internal_testing__/tiny-random-rocketqa-cross-encoder\static\inference"
res = paddle.jit.save(arg_1,arg_2,)
