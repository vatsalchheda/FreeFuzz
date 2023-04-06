import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

DATATYPE='float32'

place = fluid.CPUPlace()
#place = fluid.CUDAPlace(0)

input_data = np.array([i for i in range(1,17)]).reshape(1,1,4,4).astype(DATATYPE)
roi_data =fluid.create_lod_tensor(np.array([[1., 1., 2., 2.], [1.5, 1.5, 3., 3.]]).astype(DATATYPE),[[2]], place)
rois_num_data = np.array([2]).astype('int32')

x = fluid.data(name='input', shape=[None,1,4,4], dtype=DATATYPE)
rois = fluid.data(name='roi', shape=[None,4], dtype=DATATYPE)
rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')

pool_out = fluid.layers.roi_pool(
        input=x,
        rois=rois,
        pooled_height=1,
        pooled_width=1,
        spatial_scale=1.0,
        rois_num=rois_num)

exe = fluid.Executor(place)
out, = exe.run(feed={'input':input_data ,'roi':roi_data, 'rois_num': rois_num_data}, fetch_list=[pool_out.name])
print(out)   #array([[[[11.]]], [[[16.]]]], dtype=float32)
print(np.array(out).shape)  # (2, 1, 1, 1)