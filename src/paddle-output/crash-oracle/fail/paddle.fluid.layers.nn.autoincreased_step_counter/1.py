import paddle
arg_1 = "@LR_DECAY_COUNTER@"
arg_2 = 0
arg_3 = 1
res = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
