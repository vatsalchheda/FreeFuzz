import paddle.fluid as fluid

im_info = fluid.data(name="im_info", shape=[None, 3],
    dtype="float32")
gt_classes = fluid.data(name="gt_classes", shape=[None, 1],
    dtype="float32", lod_level=1)
is_crowd = fluid.data(name="is_crowd", shape=[None, 1],
    dtype="float32", lod_level=1)
gt_masks = fluid.data(name="gt_masks", shape=[None, 2],
    dtype="float32", lod_level=3)
# rois, roi_labels can be the output of
# fluid.layers.generate_proposal_labels.
rois = fluid.data(name="rois", shape=[None, 4],
    dtype="float32", lod_level=1)
roi_labels = fluid.data(name="roi_labels", shape=[None, 1],
    dtype="int32", lod_level=1)
mask_rois, mask_index, mask_int32 = fluid.layers.generate_mask_labels(
    im_info=im_info,
    gt_classes=gt_classes,
    is_crowd=is_crowd,
    gt_segms=gt_masks,
    rois=rois,
    labels_int32=roi_labels,
    num_classes=81,
    resolution=14)