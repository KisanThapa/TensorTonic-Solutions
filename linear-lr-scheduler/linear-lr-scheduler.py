def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.

    step:         current step (0-based)
    total_steps:  T
    initial_lr:   η0
    final_lr:     ηf
    warmup_steps: W
    """

    # if finished already
    if step > total_steps:
        return final_lr

    # warmup phase: t < W
    if warmup_steps > 0 and step < warmup_steps:
        return (step * initial_lr) / warmup_steps
    
    # decay phase: W <= t <= T
    if total_steps == warmup_steps:
        return final_lr

    return final_lr + (initial_lr - final_lr) * ((total_steps - step) / (total_steps - warmup_steps))