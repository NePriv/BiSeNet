
cfg = dict(
    model_type='bisenetv2',
    n_cats=10,  # 类别数量
    num_aux_heads=2,
    lr_start=1e-3,
    weight_decay=5e-4,
    warmup_iters=500,
    max_iter=40000,
    dataset='CustomerDatasetFloodNet',
    im_root='./datasets/floodnet',
    train_im_anns='./datasets/floodnet/train/anno.txt',
    val_im_anns='./datasets/floodnet/val/anno.txt',
    scales=[0.5, 2.0],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=4,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res_v2/floodnet',
)
