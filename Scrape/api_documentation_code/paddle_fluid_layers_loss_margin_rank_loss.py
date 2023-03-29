import paddle.fluid as fluid
label = fluid.data(name="label", shape=[-1, 1], dtype="float32")
left = fluid.data(name="left", shape=[-1, 1], dtype="float32")
right = fluid.data(name="right", shape=[-1, 1], dtype="float32")
out = fluid.layers.margin_rank_loss(label, left, right)