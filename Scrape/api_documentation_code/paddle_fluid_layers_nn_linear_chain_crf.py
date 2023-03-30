import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

#define net structure, using LodTensor
train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    input_data = fluid.data(name='input_data', shape=[-1,10], dtype='float32')
    label = fluid.data(name='label', shape=[-1,1], dtype='int')
    emission= fluid.layers.fc(input=input_data, size=10, act="tanh")
    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=label,
        param_attr=fluid.ParamAttr(
        name='crfw',
        learning_rate=0.01))
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)
#define data, using LoDTensor
a = fluid.create_lod_tensor(np.random.rand(12,10).astype('float32'), [[3,3,4,2]], place)
b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
feed1 = {'input_data':a,'label':b}
loss= exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
print(loss)

#define net structure, using padding
train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    input_data2 = fluid.data(name='input_data2', shape=[-1,10,10], dtype='float32')
    label2 = fluid.data(name='label2', shape=[-1,10,1], dtype='int')
    label_length = fluid.data(name='length', shape=[-1,1], dtype='int')
    emission2= fluid.layers.fc(input=input_data2, size=10, act="tanh", num_flatten_dims=2)
    crf_cost2 = fluid.layers.linear_chain_crf(
        input=emission2,
        label=label2,
        length=label_length,
        param_attr=fluid.ParamAttr(
         name='crfw',
         learning_rate=0.01))

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)

#define data, using padding
cc=np.random.rand(4,10,10).astype('float32')
dd=np.random.rand(4,10,1).astype('int64')
ll=np.array([[3],[3],[4],[2]])
feed2 = {'input_data2':cc,'label2':dd,'length':ll}
loss2= exe.run(train_program,feed=feed2, fetch_list=[crf_cost2])
print(loss2)
#[array([[ 7.8902354],
#        [ 7.3602567],
#        [ 10.004011],
#        [ 5.86721  ]], dtype=float32)]

#you can use find_var to get transition parameter.
transition=np.array(fluid.global_scope().find_var('crfw').get_tensor())
print(transition)