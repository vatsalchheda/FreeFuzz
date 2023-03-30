import paddle.fluid as fluid
b = fluid.default_main_program().global_block()
input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
out = fluid.layers.get_tensor_from_selected_rows(input)