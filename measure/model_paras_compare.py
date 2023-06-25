"""Tool for compare FLOPs and Params."""
import torch
import aa_nets
from thop import profile
from thop import clever_format
import warnings

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
def cal_FLOPs_and_Params(model, input):
    macs, params = profile(model, input)
    flops = macs * 2.0
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs=", flops, "Params=", params)


# ----------------------------------------------------------------------------

def cal_AAnet():
    aanet = aa_nets.AANet(192,
                          num_downsample=2,
                          feature_type='aanet',
                          no_feature_mdconv='no_feature_mdconv',
                          feature_pyramid_network='feature_pyramid_network',
                          feature_similarity='correlation',
                          aggregation_type='adaptive',
                          num_scales=3,
                          num_fusions=6,
                          num_stage_blocks=1,
                          num_deform_blocks=3,
                          no_intermediate_supervision='no_intermediate_supervision',
                          refinement_type='stereodrnet',
                          mdconv_dilation=2,
                          deformable_groups=2).to('cuda:0')
    input_left = torch.empty([1, 3, 288, 360], device='cuda:0')
    input_right = torch.empty([1, 3, 288, 360], device='cuda:0')
    cal_FLOPs_and_Params(aanet, [input_left, input_right])  # FLOPs= 58.984G Params= 3.600M


# ----------------------------------------------------------------------------
def cal_StyleGAN():
    flops, params = clever_format([1112299008, 55653], "%.3f")  # FLOPs为手动计算，可能有误
    print("styleGAN3 FLOPs=", flops, "Params=", params)
    flops, params = clever_format([3221043200, 120279], "%.3f")  # FLOPs为手动计算，可能有误
    print("styleGAN2 FLOPs=", flops, "Params=", params)
    flops, params = clever_format([3220750848, 144183], "%.3f")  # FLOPs为手动计算，可能有误
    print("styleGAN FLOPs=", flops, "Params=", params)


# ----------------------------------------------------------------------------
def cal_Diffusion_GAN():
    flops, params = clever_format([3221043200, 120020], "%.3f")  # FLOPs为手动计算，可能有误
    print("Diffusion-GAN FLOPs=", flops, "Params=", params)


# ----------------------------------------------------------------------------
def cal_TPS(k=16, h=256, w=256):
    flops, params = clever_format([2 * k * h * w, (k + 1) * h * w], "%.3f")
    print("styleGAN FLOPs=", flops, "Params=", params)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # cal_BetaVAE()  # FLOPs= 104.409G Params= 0.046M
    cal_StyleGAN()  # FLOPs= 1.112G , Params= 0.056M/0.187M(StyleGAN3)  FLOPs= 3.221G , Params = 0.144M/0.282M(StyleGAN)
    # cal_AAnet(1)  # FLOPs= 58.984G Params= 3.600M
    # cal_TPS(16)  # FLOPs= 2.097M Params= 1.114M
    # cal_TPS(25)  # FLOPs= 3.277M Params= 1.704M

# -----------------------------------------------------------------------------------
# Model                   | FLOPs(G)   | Params(M)  | MSE       |  PSNR    |  SSIM
# -----------------------------------------------------------------------------------
# 16-StyleGAN             | 3.221      | 0.144      | 63.193    | 30.124   | 0.9112
# 16-StyleGAN2            | 3.221      | 0.120      | 64.844    | 30.012   | 0.9107
# 16-Diffusion-GAN        | 3.221      | 0.120      | 64.520    | 30.034   | 0.9102
# 16-StyleGAN3            | 1.112      | 0.056      | 62.901    | 30.144   | 0.9133
# 16-TPS                  | 0.021      | 1.114      | 63.779    | 30.084   | 0.9106
# 25-TPS                  | 0.033      | 1.704      | 62.008    | 30.206   | 0.9134
# 16-BetaVAE              | 0.405      | 0.185      | 68.718    | 29.760   | 0.9004
# AANet                   | 58.984     | 3.600      | 65.139    | 29.992   | 0.9079
# -----------------------------------------------------------------------------------
