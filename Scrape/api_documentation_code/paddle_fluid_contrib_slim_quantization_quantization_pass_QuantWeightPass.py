System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/slim/quantization/quantization_pass.py:docstring of paddle.fluid.contrib.slim.quantization.quantization_pass.QuantWeightPass, line 20)
Error in “code-block” directive: maximum 1 argument(s) allowed, 22 supplied.
.. code-block:: python
    # The original graph will be rewrite.
    import paddle
    from paddle.fluid.contrib.slim.quantization                 import QuantWeightPass
    from paddle.fluid.contrib.slim.graph import IrGraph
    from paddle.fluid import core

    graph = IrGraph(core.Graph(program.desc), for_test=False)
    place = paddle.CPUPlace()
    scope = paddle.static.global_scope()
    quant_weight_pass = QuantWeightPass(scope, place)
    quant_weight_pass.apply(graph)