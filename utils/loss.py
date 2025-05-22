# @Time    : 2022/4/6 14:58
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn


"""class SoftIoULoss(nn.Module):
    def __init__(self, **kwargs):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean',
                 label_smoothing=0.0, **kwargs):
        super(CrossEntropy, self).__init__()
        self.crit = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, pred, target):
        target.squeeze(dim=1)
        loss = self.crit(pred, target)
        return loss


class BCEWithLogits(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, **kwargs):
        super(BCEWithLogits, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, pred, target):
        loss = self.crit(pred, target)
        return loss
"""


import torch.nn.functional as F
# numpy 可能在某些辅助计算中用到，但尽量使用 torch
# import numpy as np # 如果需要 np.pi, 但 torch.pi 更好用

# 你现有的 SoftIoULoss (或者一个类似的 IoU 损失基类)
# 我会稍微修改它，使其更明确地处理批量数据并返回每个样本的损失的平均值
class SoftIoULoss(nn.Module): #
    def __init__(self, smooth=1.0, **kwargs): # 添加 smooth 作为参数
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred 是模型的原始输出 (logits)，target 是真实的标签 (0或1)
        pred = torch.sigmoid(pred) #
        
        # 确保 target 和 pred 有相同的维度进行后续计算
        # 通常 pred 是 (B, 1, H, W), target 可能是 (B, 1, H, W) 或 (B, H, W)
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(1)
        
        # 计算交集和并集时，在空间维度上求和
        intersection = (pred * target).sum(dim=(1, 2, 3)) #
        pred_sum = pred.sum(dim=(1, 2, 3)) #
        target_sum = target.sum(dim=(1, 2, 3)) #
        
        # IoU loss
        loss = (intersection + self.smooth) / (pred_sum + target_sum - intersection + self.smooth) #
        loss = 1 - loss #
        
        return loss.mean() # 返回批次中所有样本损失的平均值


class SizeSensitiveLoss(nn.Module):
    """
    实现尺寸敏感损失 L_S = 1 - w * IoU
    w = (min(|A_p|, |A_gt|) + Var(|A_p|, |A_gt|)) / (max(|A_p|, |A_gt|) + Var(|A_p|, |A_gt|))
    Var(|A_p|, |A_gt|) = (|A_p| - |A_gt|)^2 / 4
    """
    def __init__(self, smooth_iou=1.0, eps=1e-6, **kwargs):
        super(SizeSensitiveLoss, self).__init__()
        self.smooth_iou = smooth_iou # 用于计算 IoU 部分的平滑因子
        self.eps = eps # 用于防止除以零的小常数

    def forward(self, pred, target):
        pred_probs = torch.sigmoid(pred) # (B, 1, H, W)

        # 确保 target 和 pred_probs 有相同的维度
        if target.dim() == pred_probs.dim() -1 :
            target = target.unsqueeze(1)
        if target.dtype != pred_probs.dtype: # 确保数据类型一致
            target = target.type_as(pred_probs)

        # 移除通道维度，因为是单类分割 (B, H, W)
        pred_probs_flat = pred_probs.squeeze(1)
        target_flat = target.squeeze(1)

        # 面积计算 |A_p| 和 |A_gt|
        area_p = torch.sum(pred_probs_flat, dim=(1, 2))  # Shape: (B)
        area_gt = torch.sum(target_flat, dim=(1, 2))     # Shape: (B)

        # 计算 IoU 部分
        intersection = (pred_probs_flat * target_flat).sum(dim=(1, 2))
        union = area_p + area_gt - intersection
        iou = (intersection + self.smooth_iou) / (union + self.smooth_iou + self.eps)

        # 方差项 Var(|A_p|, |A_gt|) = (|A_p| - |A_gt|)^2 / 4
        var_areas = ((area_p - area_gt)**2) / 4.0

        # 权重 w
        min_areas = torch.min(area_p, area_gt)
        max_areas = torch.max(area_p, area_gt)

        numerator_w = min_areas + var_areas
        denominator_w = max_areas + var_areas + self.eps # 在分母中加入 eps 防止除零

        w = numerator_w / denominator_w
        
        # 尺寸敏感损失 L_S
        loss_s = 1 - w * iou
        return loss_s.mean()


class LocationSensitiveLoss(nn.Module):
    """
    实现位置敏感损失 L_L = (1 - min(d_p, d_gt) / max(d_p, d_gt)) + (4/pi^2) * (theta_p - theta_gt)^2
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(LocationSensitiveLoss, self).__init__()
        self.eps = eps

    def _get_center(self, mask, h, w):
        # mask: (B, H, W)，可以是概率图或二值图
        # 返回: (B, 2)，对应 (cx, cy)
        
        # 创建坐标网格
        # x_coords (列索引), y_coords (行索引)
        # device=mask.device 确保张量在同一设备上
        x_indices = torch.linspace(0, w - 1, steps=w, device=mask.device).unsqueeze(0).repeat(h, 1) # (H, W)
        y_indices = torch.linspace(0, h - 1, steps=h, device=mask.device).unsqueeze(1).repeat(1, w) # (H, W)

        # 为批次扩展维度
        # (B, H, W)
        x_indices_batch = x_indices.unsqueeze(0).expand_as(mask)
        y_indices_batch = y_indices.unsqueeze(0).expand_as(mask)

        area = torch.sum(mask, dim=(1, 2)) + self.eps # (B)，加eps防止面积为0时除零

        # 坐标的加权平均
        cx = torch.sum(mask * x_indices_batch, dim=(1, 2)) / area # (B)
        cy = torch.sum(mask * y_indices_batch, dim=(1, 2)) / area # (B)
        
        return torch.stack((cx, cy), dim=1) # (B, 2)

    def forward(self, pred, target):
        pred_probs = torch.sigmoid(pred) # (B, 1, H, W)
        
        b, c, h, w = pred_probs.shape
        # 假设是单通道分割，移除通道维度
        if c == 1:
            pred_probs_flat = pred_probs.squeeze(1) # (B, H, W)
            if target.dim() == 4 and target.size(1) == 1:
                 target_flat = target.squeeze(1) # (B, H, W)
            else: # target 已经是 (B,H,W)
                 target_flat = target
        else: # 多通道情况不适用于当前损失函数定义
            raise ValueError("LocationSensitiveLoss expects single channel input.")

        if target_flat.dtype != pred_probs_flat.dtype: # 确保数据类型一致
            target_flat = target_flat.type_as(pred_probs_flat)


        # 获取中心点
        center_p = self._get_center(pred_probs_flat, h, w) # (B, 2)
        center_gt = self._get_center(target_flat, h, w)    # (B, 2)
        
        xp, yp = center_p[:, 0], center_p[:, 1]
        xgt, ygt = center_gt[:, 0], center_gt[:, 1]

        # 转换为极坐标
        # d = sqrt(x^2 + y^2)
        # theta = atan2(y, x) # atan2 结果在 [-pi, pi]
        dp = torch.sqrt(xp**2 + yp**2 + self.eps) # 加 eps 防止 sqrt(0) 可能的梯度问题
        theta_p = torch.atan2(yp, xp)

        dgt = torch.sqrt(xgt**2 + ygt**2 + self.eps)
        theta_gt = torch.atan2(ygt, xgt)

        # 位置敏感损失 L_L
        # 项1: (1 - min(d_p, d_gt) / (max(d_p, d_gt) + eps))
        min_d = torch.min(dp, dgt)
        max_d = torch.max(dp, dgt)
        term1 = 1 - (min_d / (max_d + self.eps)) # 在分母中加入 eps 防止除零
        
        # 项2: (4 / pi^2) * (theta_p - theta_gt)^2
        # 注意：角度差的平方，theta_p 和 theta_gt 的范围是 [-pi, pi]
        angle_diff = theta_p - theta_gt
        # 如果需要处理角度周期性，例如 -pi 和 pi 实际很接近，可以进行归一化
        # angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
        # 但论文中直接使用差的平方，我们先遵循论文公式
        angle_diff_sq = angle_diff**2
        term2 = (4.0 / (torch.pi**2)) * angle_diff_sq
        
        loss_l = term1 + term2
        return loss_l.mean()


class CustomCombinedLoss(nn.Module):
    def __init__(self, lambda_iou=1.0, lambda_s=1.0, lambda_l=1.0, 
                 smooth_iou=1.0, smooth_iou_s=1.0, eps_s=1e-6, eps_l=1e-6, **kwargs):
        super(CustomCombinedLoss, self).__init__()
        self.lambda_iou = lambda_iou
        self.lambda_s = lambda_s
        self.lambda_l = lambda_l

        # 实例化各个损失组件
        # SoftIoULoss
        self.iou_loss_fn = SoftIoULoss(smooth=smooth_iou, **kwargs) if self.lambda_iou > 0 else None
        self.size_loss_fn = SizeSensitiveLoss(smooth_iou=smooth_iou_s, eps=eps_s, **kwargs) if self.lambda_s > 0 else None
        self.loc_loss_fn = LocationSensitiveLoss(eps=eps_l, **kwargs) if self.lambda_l > 0 else None
        
    def forward(self, pred, target):
        loss_iou_val = torch.tensor(0.0, device=pred.device)
        loss_s_val = torch.tensor(0.0, device=pred.device)
        loss_l_val = torch.tensor(0.0, device=pred.device)

        if self.iou_loss_fn:
            loss_iou_val = self.iou_loss_fn(pred, target)
        if self.size_loss_fn:
            loss_s_val = self.size_loss_fn(pred, target)
        if self.loc_loss_fn:
            loss_l_val = self.loc_loss_fn(pred, target)
        
        total_loss = (self.lambda_iou * loss_iou_val +
                      self.lambda_s * loss_s_val +
                      self.lambda_l * loss_l_val)
        
        # 如果需要记录各个损失的值，可以在训练循环中单独计算或在这里存储
        # 例如: self.last_losses = {'iou': loss_iou_val.item(), 'size': loss_s_val.item(), 'loc': loss_l_val.item()}
        
        return total_loss

# 确保将新类添加到 __init__.py 中或者在这里被正确导入
# (如果 utils/loss.py 中有 __all__ 列表，也需要更新)