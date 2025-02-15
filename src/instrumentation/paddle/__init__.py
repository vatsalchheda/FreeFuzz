
import paddle

from .decorate_func import decorate_function
from .decorate_cls import decorate_class
import inspect

def hijack(obj, func_name_str):
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]
    module_obj = obj

    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    def is_class(x):
        return inspect.isclass(x)
    def is_callable(x):
        return callable(x)

    if is_class(orig_func):
        wrapped_func = decorate_class(orig_func, func_name_str)
    elif is_callable(orig_func):
        wrapped_func = decorate_function(orig_func, func_name_str)
    else:
        wrapped_func = orig_func
    setattr(module_obj, func_name, wrapped_func)


with open(__file__.replace("__init__.py", "paddle.txt"), "r") as f1:
    lines = f1.readlines()
    skipped = ["fluid.incubate.fleet.parameter_server.pslib.optimizer_factory.DistributedAdam", "Tensor"]
    for l in lines:
        l = l.strip()
        if l not in skipped:
            hijack(paddle, l)