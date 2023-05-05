import paddle
arg_1 = -29
arg_2 = "reflect"
arg_3 = 1
res = paddle.fluid.layers.nn.autoincreased_step_counter(counter_name=arg_1,begin=arg_2,step=arg_3,)
