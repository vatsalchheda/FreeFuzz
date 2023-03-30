import paddle.fluid as fluid
warmup_steps = 100
learning_rate = 0.01
lr = fluid.layers.learning_rate_scheduler.noam_decay(
               1/(warmup_steps *(learning_rate ** 2)),
               warmup_steps,
               learning_rate)