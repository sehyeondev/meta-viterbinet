def trainer_repr(trainer):
    return ("Trainer(\n"
            f"    run_name={trainer.run_name},\n"
            f"    use_ecc={trainer.use_ecc},\n"
            f"    n_symbols={trainer.n_symbols},\n"
            f"    memory_length={trainer.memory_length},\n"
            f"    channel_type={trainer.channel_type},\n"
            f"    channel_coefficients={trainer.channel_coefficients},\n"
            f"    noisy_est_var={trainer.noisy_est_var},\n"
            f"    fading_in_channel={trainer.fading_in_channel},\n"
            f"    fading_in_decoder={trainer.fading_in_decoder},\n"
            f"    fading_taps_type={trainer.fading_taps_type},\n"
            f"    subframes_in_frame={trainer.subframes_in_frame},\n"
            f"    gamma={trainer.gamma},\n"
            f"    val_block_length={trainer.val_block_length},\n"
            f"    val_frames={trainer.val_frames},\n"
            f"    val_SNR_start={trainer.val_SNR_start},\n"
            f"    val_SNR_end={trainer.val_SNR_end},\n"
            f"    val_SNR_step={trainer.val_SNR_step},\n"
            f"    train_block_length={trainer.train_block_length},\n"
            f"    train_frames={trainer.train_frames},\n"
            f"    train_minibatch_num={trainer.train_minibatch_num},\n"
            f"    train_minibatch_size={trainer.train_minibatch_size},\n"
            f"    train_SNR_start={trainer.train_SNR_start},\n"
            f"    train_SNR_end={trainer.train_SNR_end},\n"
            f"    train_SNR_step={trainer.train_SNR_step},\n"
            f"    lr={trainer.lr},\n"
            f"    loss_type={trainer.loss_type},\n"
            f"    optimizer_type={trainer.optimizer_type},\n"
            f"    self_supervised={trainer.self_supervised},\n"
            f"    self_supervised_iterations={trainer.self_supervised_iterations},\n"
            f"    ser_thresh={trainer.ser_thresh},\n"
            f"    meta_lr={trainer.meta_lr},\n"
            f"    MAML={trainer.MAML},\n"
            f"    online_meta={trainer.online_meta},\n"
            f"    weights_init={trainer.weights_init},\n"
            f"    window_size={trainer.window_size},\n"
            f"    buffer_empty={trainer.buffer_empty},\n"
            f"    meta_train_iterations={trainer.meta_train_iterations},\n"
            f"    meta_j_num={trainer.meta_j_num},\n"
            f"    meta_subframes={trainer.meta_subframes},\n"
            f"    noise_seed={trainer.noise_seed},\n"
            f"    word_seed={trainer.word_seed},\n"
            f"    weights_dir={trainer.weights_dir}\n"
            ")")