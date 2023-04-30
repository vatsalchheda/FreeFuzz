import paddle
arg_1 = "func"
res = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
