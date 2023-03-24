import paddle

func_name_list = ['paddle','audio','info']
func_name = func_name_list[-1]
module_obj = paddle
if len(func_name_list) > 1:
    for module_name in func_name_list[:-1]:
        module_obj = getattr(module_obj, module_name)
        print(module_obj)
orig_func = getattr(module_obj, func_name)
print(orig_func)