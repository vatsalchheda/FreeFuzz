import paddle
arg_1 = 8
arg_2 = 125
arg_3 = 40.01
res = paddle.fluid.layers.learning_rate_scheduler.noam_decay(arg_1,arg_2,arg_3,)
