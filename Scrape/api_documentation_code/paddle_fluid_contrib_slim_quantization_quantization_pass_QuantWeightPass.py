import paddle
from paddle.fluid.contrib.slim.quantization import QuantWeightPass
from paddle.fluid.contrib.slim.graph import IrGraph
from paddle.fluid import core

graph = IrGraph(core.Graph(paddle.fluid.default_main_program().desc), for_test=False)
place = paddle.CPUPlace()
scope = paddle.static.global_scope()
quant_weight_pass = QuantWeightPass(scope, place)
quant_weight_pass.apply(graph)