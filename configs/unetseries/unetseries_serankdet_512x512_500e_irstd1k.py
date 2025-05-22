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
develop = dict(source_file_root='/data1/ppw/works/All_ISTD/model/UNetSeries/SeRankDet.py')
# random_seed = 64
find_unused_parameters = True

# python rebuild_train.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py
