import paddle.fluid as fluid
import paddle

paddle.enable_static()

neg_size = 10
label = fluid.data(
          name="label", shape=[3, 1], dtype="int64")
predict = fluid.data(
          name="predict", shape=[3, neg_size + 1], dtype="float32")
cost = fluid.layers.bpr_loss(input=predict, label=label)