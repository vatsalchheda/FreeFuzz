import paddle
arg_1 = "forward"
res = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
