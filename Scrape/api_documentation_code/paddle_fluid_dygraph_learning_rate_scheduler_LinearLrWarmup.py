import paddle.fluid as fluid

learning_rate = 0.1
warmup_steps = 50
start_lr = 0
end_lr = 0.1

with fluid.dygraph.guard():
    lr_decay = fluid.dygraph.LinearLrWarmup( learning_rate, warmup_steps, start_lr, end_lr)