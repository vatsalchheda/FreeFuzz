import paddle.fluid as fluid
cuda_pinned_places_cpu_num = fluid.cuda_pinned_places()
# or
cuda_pinned_places = fluid.cuda_pinned_places(1)