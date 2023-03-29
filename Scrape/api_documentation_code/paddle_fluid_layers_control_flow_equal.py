import paddle.fluid as fluid
import numpy as np
out_cond =fluid.data(name="input1", shape=[2], dtype='bool')
label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
label_cond = fluid.layers.assign(np.array([1, 2], dtype="int32"))
out1 = fluid.layers.equal(x=label,y=limit) #out1=[True, False]
out2 = fluid.layers.equal(x=label_cond,y=limit, cond=out_cond) #out2=[False, True] out_cond=[False, True]