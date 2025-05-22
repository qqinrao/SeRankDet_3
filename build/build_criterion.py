# @Time    : 2022/9/14 20:12
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_criterion.py
# @Software: PyCharm
from utils.loss import *

"""__all__ = ['build_criterion', 'SoftIoULoss', 'BCEWithLogits', 'CrossEntropy']


#  TODO Multiple loss functions
def build_criterion(cfg):
    criterion_name = cfg.model['loss']['type']
    criterion_class = globals()[criterion_name]
    criterion = criterion_class(**cfg.model['loss'])
    return criterion"""

# 将新的损失类添加到 __all__ 列表中
__all__ = ['build_criterion', 'SoftIoULoss', 'BCEWithLogits', 'CrossEntropy',  #
           'SizeSensitiveLoss', 'LocationSensitiveLoss', 'CustomCombinedLoss']


def build_criterion(cfg): #
    criterion_name = cfg.model['loss']['type'] #
    criterion_class = globals()[criterion_name] #
    
    # 将配置文件中 cfg.model['loss'] 下的所有参数传递给损失函数的构造器
    # 在配置文件中指定 lambda_iou, lambda_s, lambda_l, smooth_iou 等参数
    criterion = criterion_class(**cfg.model['loss']) #
    return criterion
