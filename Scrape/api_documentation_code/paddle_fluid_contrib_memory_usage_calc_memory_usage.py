import paddle.fluid as fluid
lower_usage, upper_usage, unit = fluid.contrib.memory_usage(
        fluid.default_main_program(), batch_size=10)
print("memory usage is about %.3f - %.3f %s", lower_usage, upper_usage, unit)