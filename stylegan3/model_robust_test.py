# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Test robust of pretrained StyleGAN3 network pickle."""
import os
import torch
import torch.optim as optim
import aa_nets
import argparse
import numpy as np
from vae_models import BetaVAE
from utils.loss import DepthNetLoss
from utils import to_device, prepare_dataloader, load_pretrained_net
import legacy
import warnings

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
def data_transform(obstacle_sz=0, motion_deg=0, img_idx=14):
    n_img, loader = prepare_dataloader(args.data_dir, None, args.batch_size, args.num_workers, False, motion_deg)
    loss_fn = DepthNetLoss(SSIM_w=0.0, disp_gradient_w=0.0, img_w=0.0, left=args.left, upper=args.upper,
                           compensateI=args.comp).to(args.device)
    for batch, data in enumerate(loader):
        if batch != img_idx:
            continue
        # Load data
        data = to_device(data, args.device)
        left = data['left_image']
        right = data['right_image']  # shape=(N,C,H,W)

        # 遮挡测试
        x_center = 54 - 32 + 100
        y_center = 42 - 14 + 100
        mask_upper = x_center - (obstacle_sz // 2)
        mask_lower = x_center + (obstacle_sz // 2)
        mask_left = y_center - (obstacle_sz // 2)
        mask_right = y_center + (obstacle_sz // 2)
        if obstacle_sz > 0 and motion_deg == 0:
            right[:, :, mask_upper:mask_lower, mask_left:mask_right] = 0.0  # 对应区域涂黑
        return left, right, loss_fn


# ----------------------------------------------------------------------------
def train_w(args, G_model, obstacle_sz=0, motion_deg=0):
    repeat_flag = False
    z_latent = torch.zeros(args.batch_size, G_model.z_dim).to(args.device)  # 初始化z_latent
    w_latent = G_model.mapping(z_latent, None, truncation_psi=0.7, truncation_cutoff=8)
    if w_latent.dim() > 2 and w_latent.shape[1] > 1:
        repeat_flag = True
        w_latent = w_latent[:, :1, :]
    w_latent.requires_grad = True
    w_latent_last = w_latent.clone().detach()  # 上一时刻的w
    optimizer = optim.Adam([w_latent], lr=args.learning_rate, weight_decay=1.0)
    min_recons_loss = 10086.0
    left, right, loss_function = data_transform(obstacle_sz, motion_deg)
    for iter in range(args.max_iter):  # 多次迭代更新w_latent
        # 生成器
        if repeat_flag:
            w_latent_repeat = w_latent.repeat(1, 14, 1)
            disps_scale = G_model.synthesis(w_latent_repeat)
        else:
            disps_scale = G_model.synthesis(w_latent)
        disps = (disps_scale + 1) * 50.0
        loss = loss_function(disps, [left, right])
        loss_np = loss.cpu().detach().numpy()
        loss.backward(retain_graph=True)
        optimizer.step()
        w_latent_last.copy_(w_latent)

        if min_recons_loss > loss_np:
            min_recons_loss = loss_np
            disp_opt = disps.cpu().detach().numpy()
    print("min recons loss=", min_recons_loss)
    return disp_opt, min_recons_loss


# ----------------------------------------------------------------------------
def train_z(args, G_model, obstacle_sz=0, motion_deg=0):
    args.batch_size = 1
    args.learning_rate = 5e-2
    # 计算训练数据集的z均值
    data_gt = np.load("datasets/{}_disp_gt.npy".format(args.dataset))
    data_gt = data_gt / 50.0 - 1  # [-1, 1]
    data_input = torch.tensor(data_gt, dtype=torch.float32, device=args.device, requires_grad=False)
    with torch.no_grad():
        mu, log_var = G_model.encode(data_input.unsqueeze(1))
        z = G_model.reparameterize(mu, log_var)
        z_avg = torch.mean(z, dim=0).cpu().detach().numpy()
        print("z_avg=", z_avg)

    left, right, loss_function = data_transform(obstacle_sz, motion_deg)
    z_latent = torch.tensor(z_avg, dtype=torch.float32, device=args.device)
    z_latent.requires_grad = True
    optimizer = optim.Adam([z_latent], lr=args.learning_rate, weight_decay=1.0)
    min_recons_loss = 10086.0
    for iter in range(args.max_iter):
        disps_scale = G_model.decode(z_latent)
        disps = (disps_scale + 1) * 50.0
        loss = loss_function(disps, [left, right])
        loss_np = loss.cpu().detach().numpy()
        loss.backward(retain_graph=True)
        optimizer.step()

        if min_recons_loss > loss_np:
            min_recons_loss = loss_np
            disp_opt = disps.cpu().detach().numpy()

    return disp_opt, min_recons_loss


# ----------------------------------------------------------------------------
def aanet_test(args, G_model, obstacle_sz=0, motion_deg=0):
    left, right, loss_function = data_transform(obstacle_sz, motion_deg)
    disps = G_model(left, right)[-1]  # [B, H, W]
    disps = disps[:, args.upper:args.upper + 256, args.left:args.left + 256]
    loss = loss_function(disps.unsqueeze(1), [left, right])

    return disps.cpu().detach().numpy(), loss.cpu().detach().numpy()


# ----------------------------------------------------------------------------
if __name__ == "__main__":

    """
    # Test model robust at our datasets
    python stylegan3/model_robust_test.py
    
    如果报错说No module named 'utils'，在终端执行
    export PYTHONPATH=$PYTHONPATH:/home/ubuntu/G-Net/aa_nets:/home/ubuntu/G-Net/stylegan3/dnnlib:/home/ubuntu/G-Net/stylegan3/torch_utils
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='invivo', type=str,
                        help='[invivo, phantom]')
    parser.add_argument('--model', default=None, type=str,
                        help='[sgan2, sgan3, diffusion_sgan2]')
    parser.add_argument('--data_dir', default=None, type=str, help='Training dataset')
    parser.add_argument('--gt_dir', default=None, type=str, help='Training dataset gt')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers for data loading')
    parser.add_argument('--upper', default=16, type=int, help=' upper cropped coodinate, invivo:32, phantom:16')
    parser.add_argument('--left', default=70, type=int, help='left cropped coodinate, invivo:14, phantom:70')
    parser.add_argument('--comp', default=-2.0, type=float, help='light compensate, invivo:4.3, phantom:-2.0')
    parser.add_argument('--learning_rate', default=5e-2, type=float, help='Learning rate')
    parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
    parser.add_argument('--max_epoch', default=1, type=int, help='Maximum epoch number for training')
    parser.add_argument('--max_iter', default=150, type=int, help='Maximum iter number for training')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    args = parser.parse_args()

    assert args.dataset in ["invivo", "phantom"]
    if args.dataset == 'invivo':
        args.upper = 32
        args.left = 14
        args.comp = 4.3
    args.gt_dir = 'datasets/{}_disp_gt.npy'.format(args.dataset)
    assert os.path.exists(args.gt_dir)

    args.data_dir = "datasets/{}/test/".format(args.dataset)
    assert os.path.exists(args.data_dir)

    measure_list = ['motion_blur', 'obstacle']
    if args.model is None:
        model_names = ["sgan2", "diffusion_sgan2", "sgan3", "vae", "aanet"]
    else:
        model_names = [args.model]
    motion_degs = np.arange(10)
    obstacle_szs = np.arange(0, 60, 10)
    for measure in measure_list:
        if measure == 'motion_blur':
            saved_disp_data = np.zeros((len(model_names), motion_degs.size, 256, 256))
            saved_loss_data = np.zeros((len(model_names), motion_degs.size))
            for idx_m, model_name in enumerate(model_names):
                pretrained_model = 'final_models/invivo/optimizer_latest_{}.pkl'.format(model_name)  # 仅测试Invivo数据集
                assert os.path.exists(pretrained_model)
                print('=> Loading pretrained Model:', pretrained_model)
                if model_name in ["sgan2", "diffusion_sgan2", "sgan3"]:
                    with open(pretrained_model, 'rb') as f:
                        G = legacy.load_network_pkl(f)['G'].to(args.device)  # 'G_ema' for sgan3 of phantom, others 'G'
                    for idx, motion_deg in enumerate(motion_degs):
                        print("model name:", model_name, "motion_deg:", motion_deg)
                        disp_out, res_loss = train_w(args, G, motion_deg=motion_deg)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果
                elif model_name == 'vae':
                    vae = BetaVAE(1, 16).to(args.device)
                    model_path = "final_models/{}/optimizer_latest_vae.pkl".format(args.dataset)
                    assert os.path.exists(model_path)
                    load_pretrained_net(vae, model_path, no_strict=True)
                    for idx, motion_deg in enumerate(motion_degs):
                        print("model name:", model_name, "motion_deg:", motion_deg)
                        disp_out, res_loss = train_z(args, vae, motion_deg=motion_deg)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果
                else:
                    aanet = aa_nets.AANet(args.max_disp,
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
                                          deformable_groups=2).to(args.device)
                    args.pretrained_aanet = 'final_models/{}/optimizer_latest_aanet.pkl'.format(args.dataset)
                    if os.path.exists(args.pretrained_aanet):
                        print('=> Loading pretrained AANet:', args.pretrained_aanet)
                        load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
                    assert os.path.exists(args.pretrained_aanet)
                    for idx, motion_deg in enumerate(motion_degs):
                        print("model name:", model_name, "motion_deg:", motion_deg)
                        disp_out, res_loss = aanet_test(args, aanet, motion_deg=motion_deg)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果

        else:
            saved_disp_data = np.zeros((len(model_names), obstacle_szs.size, 256, 256))
            saved_loss_data = np.zeros((len(model_names), obstacle_szs.size))
            for idx_m, model_name in enumerate(model_names):
                pretrained_model = 'final_models/invivo/optimizer_latest_{}.pkl'.format(model_name)  # 仅测试Invivo数据集
                assert os.path.exists(pretrained_model)
                print('=> Loading pretrained Model:', pretrained_model)
                if model_name in ["sgan2", "diffusion_sgan2", "sgan3"]:
                    with open(pretrained_model, 'rb') as f:
                        G = legacy.load_network_pkl(f)['G'].to(args.device)  # 'G_ema' for sgan3 of phantom, others 'G'
                    for idx, obs in enumerate(obstacle_szs):
                        print("model name:", model_name, "obstacle size:", obs)
                        disp_out, res_loss = train_w(args, G, obstacle_sz=obs)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果
                elif model_name == 'vae':
                    vae = BetaVAE(1, 16).to(args.device)
                    model_path = "final_models/{}/optimizer_latest_vae.pkl".format(args.dataset)
                    assert os.path.exists(model_path)
                    load_pretrained_net(vae, model_path, no_strict=True)
                    for idx, obs in enumerate(obstacle_szs):
                        print("model name:", model_name, "obstacle size:", obs)
                        disp_out, res_loss = train_z(args, vae, obstacle_sz=obs)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果
                else:
                    aanet = aa_nets.AANet(args.max_disp,
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
                                          deformable_groups=2).to(args.device)
                    args.pretrained_aanet = 'final_models/{}/optimizer_latest_aanet.pkl'.format(args.dataset)
                    if os.path.exists(args.pretrained_aanet):
                        print('=> Loading pretrained AANet:', args.pretrained_aanet)
                        load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
                    assert os.path.exists(args.pretrained_aanet)
                    for idx, obs in enumerate(obstacle_szs):
                        print("model name:", model_name, "obstacle size:", obs)
                        disp_out, res_loss = aanet_test(args, aanet, obstacle_sz=obs)
                        saved_loss_data[idx_m][idx] = res_loss
                        saved_disp_data[idx_m][idx] = disp_out  # 统计第15帧的结果
        np.save("final_result/disp_{}_test_frame615_invivo.npy".format(measure), saved_disp_data)
        np.save("final_result/loss_{}_test_frame615_invivo.npy".format(measure), saved_loss_data)

    print("Done!")
# ----------------------------------------------------------------------------
