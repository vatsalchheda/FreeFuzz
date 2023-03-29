AdamW = fluid.contrib.extend_with_decoupled_weight_decay(
    fluid.optimizer.Adam)
optimizer = AdamW(learning_rate=0.1,
                  weight_decay=0.01)

optimizer.minimize(cost)