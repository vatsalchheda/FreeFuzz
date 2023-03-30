exe = fluid.executor(place)
map_evaluator = fluid.Evaluator.DetectionMAP(input,
    gt_label, gt_box, gt_difficult)
cur_map, accum_map = map_evaluator.get_map_var()
fetch = [cost, cur_map, accum_map]
for epoch in PASS_NUM:
    map_evaluator.reset(exe)
    for data in batches:
        loss, cur_map_v, accum_map_v = exe.run(fetch_list=fetch)