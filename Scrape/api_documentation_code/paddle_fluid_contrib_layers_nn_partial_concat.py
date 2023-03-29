System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/layers/nn.py:docstring of paddle.fluid.contrib.layers.nn.partial_concat, line 36)
Error in “code-block” directive: maximum 1 argument(s) allowed, 22 supplied.
.. code-block:: python
    import paddle.fluid as fluid
    x = fluid.data(name="x", shape=[None,3], dtype="float32")
    y = fluid.data(name="y", shape=[None,3], dtype="float32")
    concat = fluid.contrib.layers.partial_concat(
        [x, y], start_index=0, length=2)