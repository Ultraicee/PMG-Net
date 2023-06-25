import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp


def erosion2d(bin_img, kernel):
    ksize = kernel.shape[0]
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    # 将原图 unfold 成 patch，大小为核尺寸
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # 减去核值
    kernel_tensor = torch.tensor(kernel).to(patches.device)
    patches_ = torch.sub(patches, kernel_tensor)
    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches_.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                 width, 1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp / width  # Disparity is passed in NHW format
    x_shifts = x_shifts.squeeze(1)

    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3).type_as(img)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                           padding_mode='zeros')
    return output


def generate_image_left(img, disp):
    return apply_disparity(img, -disp)


def generate_image_right(img, disp):
    return apply_disparity(img, disp)


def disp_smoothness(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.abs(smoothness_x) + torch.abs(smoothness_y)


def rgb2gray(img):
    """
    NCHW->NHW
    """
    if img.ndim == 4 and img.shape[1] == 3:
        return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    elif img.ndim == 3 and img.shape[0] == 3:
        return 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]


def SSIM(x, y):
    if x.max() > 1:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
    else:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return SSIM, torch.clamp(1 - SSIM, 0, 1)


class DepthNetLoss(nn.modules.Module):
    def __init__(self, SSIM_w=0.85, disp_gradient_w=1.0, img_w=1.0, left=14, upper=32, compensateI=4.3, thres=160.0):
        super(DepthNetLoss, self).__init__()
        self.right_recons_clip = None
        self.right_clip = None
        self.ssim_valid = None
        self.loss_pho = None
        self.l1_valid = None
        self.image_loss = None
        self.compa_sum = None
        self.right_recons = None
        self.disp_gradient = None
        self.compa_mask_valid = None
        self.d_width = None
        self.d_height = None
        self.SSIM_w = SSIM_w
        self.img_w = img_w
        self.disp_gradient_w = disp_gradient_w
        self.left = left
        self.upper = upper
        self.compensateI = compensateI
        self.thres = thres

    def compute_poi_loss_pho(self, left_clip, right_clip):
        # NCHW，裁剪至poi大小
        right_recons_clip = self.right_recons[:, :, self.upper:self.d_height + self.upper,
                         self.left:self.left + self.d_width]
        est_clip_gray = rgb2gray(right_recons_clip)
        left_clip_gray = rgb2gray(left_clip)
        right_clip_gray = rgb2gray(right_clip)

        compa2 = torch.lt(est_clip_gray, self.thres)  # 重建图像的非反射区域
        compa2 = torch.unsqueeze(compa2, 1)
        compa3 = torch.lt(right_clip_gray, self.thres)  # 右图的非反射区域
        compa3 = torch.unsqueeze(compa3, 1)  # [NCHW]
        # 左图的非反射区域
        compa4 = torch.gt(left_clip_gray, self.thres)
        compa4 = torch.unsqueeze(compa4, 1)
        compa4_pad = torch.logical_not(F.pad(compa4, pad=(0, 100, 0, 0, 0, 0, 0, 0)))


        compa3_1 = compa4_pad[:, :, :, 67:67 + self.d_width]
        compa3_2 = compa4_pad[:, :, :, 72:72 + self.d_width]
        compa3_3 = compa4_pad[:, :, :, 77:77 + self.d_width]
        compa3_4 = compa4_pad[:, :, :, 82:82 + self.d_width]
        compa3_5 = compa4_pad[:, :, :, 87:87 + self.d_width]

        compa = (compa2 & compa3 & compa3_1 & compa3_2 & compa3_3 & compa3_4 & compa3_5).float()

        kernel = np.array([
            [-1, -1, 0, 0, 0, -1, -1],
            [-1, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, -1],
            [-1, -1, 0, 0, 0, -1, -1]]).astype(np.float32)

        compa_erosion = erosion2d(compa, kernel)
        compa_rgb = torch.tile(compa_erosion, [1, 3, 1, 1])

        self.compa_mask_valid = compa_rgb
        loss_diff = right_recons_clip - right_clip - self.compensateI
        loss_pho_sum = torch.sum(
            torch.mul(torch.square(loss_diff), compa_rgb), dim=(1, 2, 3))
        compa_sum = torch.sum(compa_rgb, dim=(1, 2, 3))
        self.compa_sum = compa_sum
        loss_pho = torch.mean(torch.divide(loss_pho_sum, compa_sum))
        return loss_pho

    def forward(self, input, target):
        """
        Args:
            input [disp]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        height, width = left.shape[-2], left.shape[-1]
        self.d_height, self.d_width = input.shape[-2], input.shape[-1]  # NCHW
        # Prepare disparities

        disp_right = F.pad(input,
                               (self.left, width - self.d_width - self.left, self.upper,
                                height - self.d_height - self.upper),
                               'constant',
                               0.0)

        # Generate images
        self.right_recons = generate_image_right(left, disp_right)
        left_clip = left[:, :, self.upper:self.d_height + self.upper, self.left:self.left + self.d_width]
        right_clip = right[:, :, self.upper:self.d_height + self.upper, self.left:self.left + self.d_width]
        right_recons_clip = self.right_recons[:, :, self.upper:self.d_height + self.upper, self.left:self.left + self.d_width]

        # Reconstruction loss
        loss_pho = self.compute_poi_loss_pho(left_clip, right_clip)  # 得到self.compa_mask_valid
        self.loss_pho = loss_pho

        # Disparities smoothness
        right_clip_valid = torch.mul(right_clip + self.compensateI, self.compa_mask_valid)
        right_recons_valid = torch.mul(right_recons_clip, self.compa_mask_valid)
        self.disp_gradient = disp_smoothness(right_clip_valid, right_recons_valid)
        loss_disp_gradient = torch.mean(torch.divide(torch.sum(torch.abs(self.disp_gradient)), self.compa_sum))

        # L1，单像素单通道光度损失
        self.l1_valid = right_recons_valid - right_clip_valid
        l1_right_loss = torch.sum(torch.abs(self.l1_valid)) / torch.sum(self.compa_sum)
        # SSIM，相似一致性损失
        ssim_valid, ssim_loss = SSIM(right_recons_valid, right_clip_valid)
        ssim_valid_mask = torch.lt(ssim_valid, 1.0)  # 置0区域相似度为1.0,需要去掉
        self.ssim_valid = torch.mean(ssim_valid[ssim_valid_mask])
        ssim_right = torch.mean(ssim_loss[ssim_valid_mask])
        loss_image = self.SSIM_w * ssim_right + (1 - self.SSIM_w) * l1_right_loss
        self.image_loss = loss_image
        loss = self.img_w * loss_image + self.disp_gradient_w * loss_disp_gradient + loss_pho
        # print("loss_image=", loss_image.cpu().detach().numpy(),
        #       "loss_disp_gradient=", loss_disp_gradient.cpu().detach().numpy(),
        #       "loss_pho=", loss_pho.cpu().detach().numpy())
        return loss
