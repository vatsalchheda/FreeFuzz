System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/layers/nn.py:docstring of paddle.fluid.contrib.layers.nn.batch_fc, line 18)
Error in “code-block” directive: maximum 1 argument(s) allowed, 5 supplied.
.. code-block:: python
   import paddle.fluid as fluid

   input = fluid.data(name="input", shape=[16, 2, 3], dtype="float32")
   out = fluid.contrib.layers.batch_fc(input=input,
                                       param_size=[16, 3, 10],
                                       param_attr=
                                         fluid.ParamAttr(learning_rate=1.0,
                                                       name="w_0",
                                                       initializer=
                                                       fluid.initializer.Xavier(uniform=False)),
                                       bias_size=[16, 10],
                                       bias_attr=
                                         fluid.ParamAttr(learning_rate=1.0,
                                                       name="b_0",
                                                       initializer=
                                                       fluid.initializer.Xavier(uniform=False)),
                                           act="relu")