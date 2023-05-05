import paddle
arg_1 = "max"
arg_2 = 100
arg_3 = 0.01
res = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
