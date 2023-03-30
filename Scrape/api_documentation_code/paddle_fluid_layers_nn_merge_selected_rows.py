import paddle.fluid as fluid
b = fluid.default_main_program().global_block()
var = b.create_var(
    name="X", dtype="float32", persistable=True,
    type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
y = fluid.layers.merge_selected_rows(var)