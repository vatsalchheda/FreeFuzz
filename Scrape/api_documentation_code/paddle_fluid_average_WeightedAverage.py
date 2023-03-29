import paddle.fluid as fluid
avg = fluid.average.WeightedAverage()
avg.add(value=2.0, weight=1)
avg.add(value=4.0, weight=2)
avg.eval()

# The result is 3.333333333.
# For (2.0 * 1 + 4.0 * 2) / (1 + 2) = 3.333333333