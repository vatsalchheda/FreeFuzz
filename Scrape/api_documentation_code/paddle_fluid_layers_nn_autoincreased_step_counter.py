import paddle.fluid as fluid
import paddle
paddle.enable_static()
global_step = fluid.layers.autoincreased_step_counter(
    counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)