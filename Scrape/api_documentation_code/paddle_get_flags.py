import paddle

flags = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
res = paddle.get_flags(flags)
print(res)
# {'FLAGS_eager_delete_tensor_gb': 0.0, 'FLAGS_check_nan_inf': False}