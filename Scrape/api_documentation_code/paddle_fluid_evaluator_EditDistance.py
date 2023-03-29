exe = fluid.executor(place)
distance_evaluator = fluid.Evaluator.EditDistance(input, label)
for epoch in PASS_NUM:
    distance_evaluator.reset(exe)
    for data in batches:
        loss = exe.run(fetch_list=[cost])
    distance, instance_error = distance_evaluator.eval(exe)