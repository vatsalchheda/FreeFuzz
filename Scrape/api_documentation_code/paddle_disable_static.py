import paddle
print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

paddle.enable_static()
print(paddle.in_dynamic_mode())  # False, Now we are in static mode

paddle.disable_static()
print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode