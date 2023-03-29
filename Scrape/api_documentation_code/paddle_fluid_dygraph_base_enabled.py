import paddle.fluid as fluid

fluid.enable_dygraph()  # Now we are in dygragh mode
print(fluid.dygraph.enabled())  # True
fluid.disable_dygraph()
print(fluid.dygraph.enabled())  # False