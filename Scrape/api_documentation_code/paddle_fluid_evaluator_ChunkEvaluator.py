exe = fluid.executor(place)
evaluator = fluid.Evaluator.ChunkEvaluator(input, label)
for epoch in PASS_NUM:
    evaluator.reset(exe)
    for data in batches:
        loss = exe.run(fetch_list=[cost])
    distance, instance_error = distance_evaluator.eval(exe)