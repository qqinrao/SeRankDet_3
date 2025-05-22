_base_ = [
    '../_base_/datasets/irstd1k.py',#数据集
    '../_base_/default_runtime.py',#运行时
    '../_base_/schedules/schedule_500e.py',#训练
    '../_base_/models/unetseries.py'#模型基础配置
]
# 覆盖unetseries_serankdet_512x512_500e_irstd1k中的 model.decode_head
model = dict(
    decode_head=dict(
        type='SeRankDet'
    ),
    # 更新损失函数配置(添加部分)
    loss=dict(
        type='CustomCombinedLoss',  # 指定使用新的组合损失函数
        lambda_iou=1.0,             # IoU 损失的权重
        lambda_s=0.5,               # 尺寸敏感损失的权重 (根据需要调整)
        lambda_l=0.5,               # 位置敏感损失的权重 (根据需要调整)
        smooth_iou=1.0,             # SoftIoULoss 组件的平滑因子
        smooth_iou_s=1.0,           # SizeSensitiveLoss 中 IoU 部分的平滑因子
        eps_s=1e-6,                 # SizeSensitiveLoss 中的 epsilon
        eps_l=1e-6                  # LocationSensitiveLoss 中的 epsilon
        # **kwargs 中其他参数会传递给损失函数构造器，
        # 例如，如果你的 SoftIoULoss 或其他基类需要特定参数，可以在这里添加
    )
)

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
)

runner = dict(type='EpochBasedRunner', max_epochs=250)
data = dict(
    train_batch=4,
    test_batch=4)
#develop = dict(source_file_root='/data1/ppw/works/All_ISTD/model/UNetSeries/SeRankDet.py')
develop = dict(source_file_root='model/SeRankDet/SeRankDet.py')
# random_seed = 64
find_unused_parameters = True

# python rebuild_train.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py
