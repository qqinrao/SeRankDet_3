from mmcv import Config

cfg = Config(dict(a=1,b=dict(b1=[0,1])))
print(cfg.b.b1)