import paddle.fluid as fluid
import paddle
paddle.enable_static()
batch_size = 64
label = fluid.data(
          name="label", shape=[batch_size, 1], dtype="int64")
similarity = fluid.data(
          name="similarity", shape=[batch_size, 1], dtype="float32")
cost = fluid.layers.teacher_student_sigmoid_loss(input=similarity, label=label)