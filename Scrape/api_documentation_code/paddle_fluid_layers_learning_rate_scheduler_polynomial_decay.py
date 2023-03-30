import paddle.fluid as fluid
start_lr = 0.01
total_step = 5000
end_lr = 0
lr = fluid.layers.polynomial_decay(
    start_lr, total_step, end_lr, power=1)