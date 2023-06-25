# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train w latents using pretrained K-StyleGAN3 network pickle."""
import os
import time
import torch
from torch import optim, nn
import argparse
import numpy as np
from utils.loss import DepthNetLoss

from utils import to_device
from dataloader.dataloader import prepare_dataloader
import legacy
import warnings

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
def adjust_learning_rate(optimizer, step, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 50 epochs after 50 steps"""

    if 50 <= step < 100:
        lr = learning_rate / 2
    elif step >= 100:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ----------------------------------------------------------------------------
def train_w(args, G_model):
    """
    无监督训练潜在编码向量w
    :param args:
    :param G_model: 生成器模型
    """

    n_img, loader = prepare_dataloader(args.data_dir, args.gt_dir, args.batch_size, args.num_workers, False)
    # data to saved
    w_out = np.zeros((100, 1, G_model.w_dim))  # 保存预测潜在向量结果[N, 1, w_dim]
    disp_out = np.zeros((100, args.poi_size, args.poi_size))  # 保存预测视差结果[N, h, w]
    min_loss_list = []
    min_loss_pho_list = []
    cost_step_list = []

    loss_function = DepthNetLoss(left=args.left, upper=args.upper,
                                 compensateI=args.comp).to(args.device)
    repeat_flag = False
    running_loss = 0.0
    epoch_start_time = time.time()

    for batch, data in enumerate(loader):
        # Load data
        data = to_device(data, args.device)
        left = data['left_image']
        right = data['right_image']

        # 固定位置初始化w
        z_latent = torch.zeros(args.batch_size, G_model.z_dim).to(args.device)  # 初始化z_latent
        w_latent = G_model.mapping(z_latent, None, truncation_psi=0.9, truncation_cutoff=8)

        # 处理不同模型之间有无复制14遍的差异
        if w_latent.dim() > 2 and w_latent.shape[1] > 1:
            repeat_flag = True
            w_latent = w_latent[:, :1, :]
        w_latent.requires_grad = True

        optimizer = optim.Adam([w_latent], lr=args.learning_rate, weight_decay=1.0)  # 设置正则化项，防止过拟合
        min_loss_pho = 10086.0
        min_loss = 10086.0
        cost_step = 0

        for iter in range(args.max_iter):  # 多次迭代更新w_latent
            # 生成器
            if repeat_flag:
                w_latent_repeat = w_latent.repeat(1, 14, 1)
                disps_scale = G_model.synthesis(w_latent_repeat)
            else:
                disps_scale = G_model.synthesis(w_latent)

            disps = 50.0 * (disps_scale + 1)

            loss = loss_function(disps,
                                 [left, right])

            loss_pho_np = loss_function.loss_pho.cpu().detach().numpy()
            loss_np = loss.cpu().detach().numpy()
            running_loss += loss_np

            loss.backward(retain_graph=True)
            optimizer.step()
            adjust_learning_rate(optimizer, iter, args.learning_rate)

            if min_loss > loss_np:
                min_loss = loss_np
                min_loss_pho = loss_pho_np
                cost_step = iter
                w_opt = w_latent.cpu().detach().numpy()
                disp_opt = disps.cpu().detach().numpy()

        # 保存每批次w结果
        min_loss_pho_list.append(min_loss_pho)
        min_loss_list.append(min_loss)
        cost_step_list.append(cost_step + 1)
        w_out[batch * args.batch_size:(batch + 1) * args.batch_size] = w_opt

        print("{} | loss_pho {:.5f} | loss {:.5f} | cost step {}"
              .format(batch + 1, min_loss_pho, min_loss, cost_step + 1))

    epoch_end_time = time.time()
    epoch_cost_time = epoch_end_time - epoch_start_time
    m = int(epoch_cost_time) // 60
    s = int(epoch_cost_time) % 60
    print("Epoch[{}/{}]".format(1, args.max_epoch),
          " loss_pho {:.5f}".format(sum(min_loss_pho_list) / len(min_loss_pho_list)),
          " cost_time {}m{}s".format(int(m), int(s)),
          " cost_step_avg {:.2f}".format(sum(cost_step_list) / len(cost_step_list)),
          )
    # 保存数据
    # res_loss = np.array(min_loss_list)
    # np.save("../datasets/w_latent_{}_{}_s.npy".format(n_img, args.model), w_out)
    # res_loss = np.array(min_loss_list)
    # np.save("final_result/loss_{}_test_p.npy".format(args.model), res_loss)


# ----------------------------------------------------------------------------
# 测试训练的disp结果
def test_disp(args):
    model_names = ["sgan", "sgan2", "sgan3"]  # change name list to test specified disparity result
    for model_name in model_names:
        print("=========== Model {} ===========".format(model_name))
        data_gt = np.load('final_result/{}/disp_{}_test_{}.npy'.format(args.dataset, model_name, args.dataset[0]))
        if data_gt.shape[-2] > args.poi_size:
            data_gt = np.squeeze(data_gt)[:, args.upper:args.upper + args.poi_size, args.left:args.left + args.poi_size]
        assert data_gt.shape[-1] == args.poi_size
        n_img, loader = prepare_dataloader(args.data_dir, None, args.batch_size, args.num_workers, False)
        loss_function = DepthNetLoss(SSIM_w=0, disp_gradient_w=0, img_w=0,
                                     left=args.left, upper=args.upper, compensateI=args.comp).to(args.device)
        loss_recons_avg = []
        epoch_start_time = time.time()
        for batch, data in enumerate(loader):
            # Load data
            data = to_device(data, args.device)
            left = data['left_image']
            right = data['right_image']
            target = torch.tensor(data_gt[batch:batch + 1, :, :], dtype=torch.float32, device=args.device,
                                  requires_grad=False)
            loss = loss_function(target.unsqueeze(1), [left, right])
            loss_recons_avg.append(loss.cpu().numpy())

            # # 保存结果
            # print("batch{} | recons_loss {}".format(batch + 1, loss.item()))
        epoch_end_time = time.time()
        epoch_cost_time = epoch_end_time - epoch_start_time
        m = int(epoch_cost_time) // 60
        s = int(epoch_cost_time) % 60
        print("Epoch[{}/{}]".format(1, args.max_epoch),
              " Avg loss_recons {:.5f}".format(sum(loss_recons_avg) / len(loss_recons_avg)),
              " cost_time {}m{}s".format(int(m), int(s))
              )
        # np.save("final_result/loss_{}_test_p.npy".format(model_name), res_loss)
        # print("=> Save recons loss to file:", "final_result/loss_{}_test_p.npy\n".format(model_name))


# ----------------------------------------------------------------------------
if __name__ == "__main__":

    """Train w using trained SGAN model at our datasets

        Examples:

        \b
        # Train w for invivo.
        python stylegan3/GNet_train.py --dataset=invivo --model=sgan3 --mode=train_w

        \b
        # Test saved disp file for invivo.
        python stylegan3/GNet_train.py --dataset=invivo --mode=test_disp

        or set default params below.
        Note: 如果报错 No module named 'utils'，在终端执行
        export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PMG-Net/aa_nets:/home/ubuntu/PMG-Net/stylegan3/dnnlib:/home/ubuntu/PMG-Net/stylegan3/torch_utils
        添加aanet和stylegan3所需的依赖库到环境变量中
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synth_invivo', type=str,
                        help='[invivo, synth_invivo, phantom]')
    parser.add_argument('--mode', default='train_w', type=str,
                        help='[train_w, test_disp]')
    parser.add_argument('--model', default='sgan3', type=str,
                        help='[sgan2, sgan3, diffusion_sgan2]')
    parser.add_argument('--data_dir', default=None, type=str, help='Training dataset')
    parser.add_argument('--gt_dir', default=None, type=str, help='Training dataset gt')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--poi_size', default=None, type=int, help='Estimated POI size')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
    parser.add_argument('--upper', default=42, type=int, help=' upper cropped coodinate, invivo:32, phantom:16')
    parser.add_argument('--left', default=37, type=int, help='left cropped coodinate, invivo:14, phantom:70')
    parser.add_argument('--comp', default=0, type=float, help='light compensate, invivo:4.3, phantom:-2.0')
    parser.add_argument('--learning_rate', default=5e-2, type=float, help='Learning rate')
    parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
    parser.add_argument('--max_epoch', default=1, type=int, help='Maximum epoch number for training')
    parser.add_argument('--max_iter', default=150, type=int, help='Maximum iter number for training')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    args = parser.parse_args()

    assert args.dataset in ["invivo", "phantom", "synth_invivo"]
    if args.dataset == "invivo":
        args.upper = 32
        args.left = 14
        args.comp = 4.3
        args.poi_size = 256
    elif args.dataset == "phantom":
        args.upper = 16
        args.left = 70
        args.comp = -2.0
        args.poi_size = 256
    else:
        # synth invivo
        args.upper = 42
        args.left = 37
        args.comp = 0
        args.poi_size = 128

    # change 'test' to 'train' when generate w_opt^{train}, to 'robust_test' when test robustness.
    args.data_dir = "datasets/{}/test/".format(args.dataset)
    assert os.path.exists(args.data_dir)

    if args.mode == "train_w" or args.mode == "0":
        # gt for test
        pretrained_styleGAN = 'final_models/{}/optimizer_latest_{}.pkl'.format(args.dataset, args.model)
        assert os.path.exists(pretrained_styleGAN)
        print('=> Loading pretrained StyleGAN:', pretrained_styleGAN)
        with open(pretrained_styleGAN, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(args.device)  # 'G_ema' for sgan3 of phantom, others 'G'
        train_w(args, G)
    elif args.mode == "test_disp" or args.mode == "1":
        test_disp(args)

    print("Done!")
# ----------------------------------------------------------------------------
