
cfg = dict(
    model_type='bisenetv1',
    n_cats=6,  # 类别数量
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    dataset='CustomerDataset',
    im_root='./datasets/vaihingen',
    train_im_anns='./datasets/anno.txt',
    val_im_anns='./datasets/anno.txt',
    scales=[0.75, 2.],
    cropsize=[224, 224],
    eval_crop=[224, 224],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
