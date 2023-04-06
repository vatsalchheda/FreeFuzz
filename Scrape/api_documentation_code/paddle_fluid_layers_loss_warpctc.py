# using Tensor
import paddle
import paddle.fluid as fluid
import numpy as np

# length of the longest logit sequence
max_seq_length = 5
#length of the longest label sequence
max_label_length = 3
# number of logit sequences
batch_size = 16
# class num
class_num = 5
paddle.enable_static()
logits = fluid.data(name='logits',
               shape=[max_seq_length, batch_size, class_num+1],
               dtype='float32')
logits_length = fluid.data(name='logits_length', shape=[None],
                 dtype='int64')
label = fluid.data(name='label', shape=[batch_size, max_label_length],
               dtype='int32')
label_length = fluid.data(name='labels_length', shape=[None],
                 dtype='int64')
cost = fluid.layers.warpctc(input=logits, label=label,
                input_length=logits_length,
                label_length=label_length)
place = fluid.CPUPlace()
x = np.random.rand(max_seq_length, batch_size, class_num+1).astype("float32")
y = np.random.randint(0, class_num, [batch_size, max_label_length]).astype("int32")
exe = fluid.Executor(place)
output= exe.run(fluid.default_main_program(),
                feed={"logits": x,
                      "label": y,
                      "logits_length": np.array([max_seq_length]*batch_size).astype("int64"),
                      "labels_length": np.array([max_label_length]*batch_size).astype("int64")},
                      fetch_list=[cost.name])
print(output)