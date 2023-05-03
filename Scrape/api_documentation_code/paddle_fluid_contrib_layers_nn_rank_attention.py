import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy as np

input = fluid.data(name="input", shape=[None, 2], dtype="float32")
rank_offset = fluid.data(name="rank_offset", shape=[None, 7], dtype="int32")
out = fluid.contrib.layers.rank_attention(input=input,
                                          rank_offset=rank_offset,
                                          rank_param_shape=[18,3],
                                          rank_param_attr=
                                            fluid.ParamAttr(learning_rate=1.0,
                                                          name="ubm_rank_param.w_0",
                                                          initializer=
                                                          fluid.initializer.Xavier(uniform=False)),
                                          max_rank=3,
                                          max_size=0)