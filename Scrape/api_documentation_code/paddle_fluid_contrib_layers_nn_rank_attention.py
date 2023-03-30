System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/layers/nn.py:docstring of paddle.fluid.contrib.layers.nn.rank_attention, line 17)
Error in “code-block” directive: maximum 1 argument(s) allowed, 9 supplied.
.. code-block:: python
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