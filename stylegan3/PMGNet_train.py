# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train Predicting-Mapping Network using trained w latents."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import legacy
import aa_nets

from torch.nn import init
from utils import to_device, load_pretrained_net
from dataloader.dataloader import prepare_dataloader
from utils.loss import DepthNetLoss

import warnings

warnings.filterwarnings("ignore")


# 在终端执行cmd
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PMG-Net/aa_nets:/home/ubuntu/PMG-Net/stylegan3/dnnlib:/home/ubuntu/PMG-Net/stylegan3/torch_utils


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
class GRU(nn.Module):
    """ GRU module"""
    def __init__(self, input_dim, hidden_num, output_dim, layer_num=2, use_cuda=True):
        super(GRU, self).__init__()
        self.use_cuda = use_cuda
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_num,
            num_layers=layer_num,
            dropout=0.1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_num, output_dim)
        optim_range = np.sqrt(1.0 / hidden_num)
        self.weightInit(optim_range)

    def forward(self, x):
        h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_num).requires_grad_()
        h0 = h0.cuda() if self.use_cuda else h0
        gru_out, _ = self.gru(x, h0.detach())
        p_out = self.out(gru_out)
        return p_out

    def weightInit(self, gain=1):
        for name, param in self.named_parameters():
            if 'gru.weight' in name:
                init.orthogonal_(param, gain)

# ----------------------------------------------------------------------------
class PMG(nn.Module):
    def __init__(self, G, args):
        super(PMG, self).__init__()
        self.Gs = G
        self.Ms = aa_nets.AANet(192,
                                num_downsample=2,
                                feature_type='aanet',
                                no_feature_mdconv='no_feature_mdconv',
                                feature_pyramid_network='feature_pyramid_network',
                                feature_similarity='correlation',
                                aggregation_type='adaptive',
                                num_scales=1,
                                num_fusions=6,
                                num_stage_blocks=1,
                                num_deform_blocks=3,
                                no_intermediate_supervision='no_intermediate_supervision',
                                refinement_type='stereodrnet',
                                mdconv_dilation=2,
                                deformable_groups=2)

        self.latent_dim = G.w_dim
        self.left = args.left//3
        self.upper = args.upper//3
        self.model = args.model

        self.Ps = GRU(input_dim=self.latent_dim * 2, hidden_num=300, output_dim=self.latent_dim, layer_num=2)
        # Process cost volume
        self.refine = nn.Sequential(
            nn.Conv2d(64, out_channels=32,
                      kernel_size=3, stride=1, padding=1, dilation=2),
            nn.Conv2d(32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, out_channels=16,
                      kernel_size=3, stride=1, padding=1, dilation=2),
            nn.Conv2d(16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        # Build Mapping
        self.map = nn.Sequential(
            nn.Linear(6400, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, self.latent_dim),
        )

    def forward(self, x, latent=None):
        # AANet中的提取提取和代价聚合
        left = x['left_image']/255.0
        right = x['right_image']/255.0
        left_feature = self.Ms.feature_extraction(left)
        right_feature = self.Ms.feature_extraction(right)
        cost_volume = self.Ms.cost_volume_construction(left_feature, right_feature)
        aggregation = self.Ms.aggregation(cost_volume)
        # 训练精炼模块
        x = self.refine(aggregation[0][:, :, self.upper:86+self.upper, self.left:86+self.left])  # 裁剪有效区域
        x = torch.flatten(x, start_dim=1)
        # 训练映射网络
        x = self.map(x)
        # 额外训练预测器
        if latent is not None:
            x_in = torch.cat((x, latent), 1)
            x = self.Ps(x_in.unsqueeze(1))
        return x

# ----------------------------------------------------------------------------
class SiamG(nn.Module):
    """Siamese Networks with VGG module"""
    def __init__(self, G, args):
        super(SiamG, self).__init__()
        self.Gs = G
        self.latent_dim = G.w_dim
        self.left = args.left
        self.upper = args.upper
        # Build Feature extractor
        modules = []
        in_channels = 3
        hidden_dims = [64, 128, 64, 32]
        dilation_sz = [2, 2, 1, 1]
        # 第一层的第二层卷积stride=1
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=32,
                          kernel_size=3, stride=1, padding=1, dilation=2),
                nn.Conv2d(32, out_channels=32,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )
        # 其他层的第二层卷积stride=2
        in_channels = 32
        for h_dim, d_sz in zip(hidden_dims, dilation_sz):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1, dilation=d_sz),
                    nn.Conv2d(h_dim, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    # nn.Dropout(p=0.5)
                )
            )
            in_channels = h_dim

        self.feature_extractor = nn.Sequential(*modules)
        self.mapping_final = nn.Sequential(
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.5),
            nn.Linear(256, self.Gs.w_dim),
        )

    def forward(self, x):
        left = x['left_image'][:, :, self.upper:self.upper+256, self.left:self.left+256]/255
        right = x['right_image'][:, :, self.upper:self.upper+256, self.left:self.left+256]/255
        left = self.feature_extractor(left)
        right = self.feature_extractor(right)
        x = torch.cat([left, right], dim=1)  # 拼接channel
        x = torch.flatten(x, start_dim=1)
        x = self.mapping_final(x)

        return x

# ----------------------------------------------------------------------------
def train(args, model):
    loss_fn = nn.MSELoss()
    model.train()
    args.max_epoch = 20
    args.batch_size = 5
    shuffle_flag = True

    if args.model == 'mapping':
        optimizer = optim.Adam([{"params": model.feature_extractor.parameters()},
                                {"params": model.mapping_final.parameters()}],
                               lr=1e-4, weight_decay=0.0)
    elif args.model == 'mg':
        optimizer = optim.Adam([{"params": model.Ms.parameters()},
                                {"params": model.refine.parameters()},
                                {"params": model.map.parameters()}],
                               lr=1e-4, weight_decay=0.0)
    else:
        # pmg item, 1 by 1 and close shuffle
        args.batch_size = 1
        shuffle_flag = False

        optimizer = optim.Adam([{"params": model.Ms.parameters()},
                                {"params": model.refine.parameters()},
                                {"params": model.map.parameters()},
                                {"params": model.Ps.parameters()}],
                               lr=1e-3, weight_decay=0.0)
    # 加载数据集
    n_img, loader = prepare_dataloader(args.data_dir,
                                       'datasets/w_latent600_sgan3_{}.npy'.format(args.dataset),
                                       args.batch_size,
                                       args.num_workers,
                                       shuffle_flag)

    train_batch_num = int(5 * n_img / (6 * args.batch_size))  # training:validation=5:1
    min_epoch_loss = 1.0
    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        pmg_train_loss = 0.0
        w_last = None
        for batch, data in enumerate(loader):
            # Load data
            data = to_device(data, args.device)
            w_label = data['gt'].float().view(-1, 16)
            if args.model == 'pmg' and batch == 0:
                w_last = data['gt'].float().view(-1, 16)  # 获取第一个w
                continue

            if batch < train_batch_num:
                # training part for PMG-Net
                if args.model == 'pmg':
                    w_out = model(data, w_last)
                    w_last = data['gt'].float().view(-1, 16)  # 获取下一batch的w_opt
                    loss = loss_fn(w_out, w_label)
                    pmg_train_loss += loss
                    loss_curr, current = loss.item(), (batch + 1)

                    if batch % 5 == 0:
                        # batch update
                        optimizer.zero_grad()
                        pmg_train_loss.backward()
                        optimizer.step()
                        pmg_train_loss = 0.0
                else:
                    # training part for MG-Net and Siam-Net
                    w_out = model(data)
                    loss = loss_fn(w_out, w_label)
                    pmg_train_loss += loss
                    loss_curr, current = loss.item(), (batch + 1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # print(f"training loss: {loss_curr:>5f}  [{current:>3d}/{train_batch_num:>3d}]")
            else:
                # validation part
                with torch.no_grad():
                    if args.model == 'pmg':
                        w_out = model(data, w_last)
                        w_last = data['gt'].float().view(-1, 16)  # 获取当前batch的w用作下一batch
                    else:
                        w_out = model(data)
                loss = loss_fn(w_out, w_label)
                loss_curr, current = loss.item(), (batch + 1)
                epoch_loss += loss.cpu().detach().numpy().item()
                # print(f"valid loss: {loss_curr:>5f}  [{current - train_batch_num:>3d}/{int(train_batch_num/6):>3d}]")

        epoch_end_time = time.time()
        epoch_cost_time = epoch_end_time - epoch_start_time
        m = int(epoch_cost_time) // 60
        s = int(epoch_cost_time) % 60
        curr_ave = epoch_loss / int(train_batch_num/5)  # training:validation=5:1
        print("Epoch[{}/{}]".format(epoch + 1, args.max_epoch),
              " loss ave {:.5f}".format(curr_ave),
              " cost time {}m{}s".format(int(m), int(s))
              )

        # Save model
        if args.data_output:
            if not os.path.exists(args.data_output):
                os.makedirs(args.data_output)
            PATH = args.data_output + "/optimizer_latest_{}.pkl".format(args.model)
            if curr_ave < min_epoch_loss:
                min_epoch_loss = curr_ave
                print("=> Save model:%s" % PATH)
                torch.save(model.state_dict(), PATH)


# ----------------------------------------------------------------------------
def test(args, G_model):
    if os.path.exists("final_models/{}/optimizer_latest_{}.pkl".format(args.dataset, args.model)):
        print("Use model file from: optimizer_latest_{}.pkl".format(args.model))
        model.load_state_dict(torch.load("final_models/{}/optimizer_latest_{}.pkl".format(args.dataset, args.model)))
        model.eval()  # 关闭BN、Dorpout

    args.batch_size = 1
    args.learning_rate = 5e-2
    n_img, loader = prepare_dataloader(args.data_dir, None, args.batch_size, args.num_workers, False)
    w_out = np.zeros((n_img, 1, G_model.w_dim))  # 保存预测结果[N, 1, w_dim]
    disp_out = np.zeros((n_img, 256, 256))  # 保存预测结果[N, h, w]
    min_loss_list = []
    cost_step_list = []
    P_total_times = 0.0
    G_total_times = 0.0
    loss_function = DepthNetLoss(SSIM_w=0.0, disp_gradient_w=0.0, img_w=0.0, left=args.left, upper=args.upper,
                                 compensateI=args.comp).to(args.device)
    repeat_flag = False
    running_loss = 0.0
    epoch_start_time = time.time()
    if args.model == 'mean':
        # 计算w_avg
        w_avg = np.mean(np.load("/home/ubuntu/WS-YG/PMG-Net/datasets/w_latent600_sgan3_{}.npy".format(args.dataset)), axis=0)
        print("w_opt's w_avg=\n", w_avg)
    for batch, data in enumerate(loader):
        # Load data
        data = to_device(data, args.device)
        left = data['left_image']
        right = data['right_image']
        batch_start_time = time.time()
        if args.model in ['pmg', 'mg', 'mapping']:
            with torch.no_grad():
                z_latent = torch.zeros(args.batch_size, G_model.z_dim).to(args.device)
                w_latent = model(data, z_latent)
        elif args.model == 'last':
            z_latent = torch.zeros(args.batch_size, G_model.z_dim).to(args.device)  # 初始化z_latent
            w_latent = G_model.mapping(z_latent, None, truncation_psi=0.7, truncation_cutoff=8)
        else:
            w_latent = torch.tensor(w_avg, dtype=torch.float32, device=args.device)  # 使用均值作为起点
        if w_latent.dim() < 3:
            w_latent = w_latent.unsqueeze(1)
        if w_latent.shape[1] > 1:
            repeat_flag = True
            w_latent = w_latent[:, :1, :]

        w_latent.requires_grad = True
        optimizer = optim.Adam([w_latent], lr=args.learning_rate, weight_decay=0.0)
        min_recons_loss = 10086.0
        cost_step = 0
        batch_end_time = time.time()
        P_total_times = (batch_end_time-batch_start_time)
        iter_start_time = time.time()
        for step in range(args.max_step):  # 多次迭代更新w_latent
            # 生成器
            if repeat_flag:
                w_latent_repeat = w_latent.repeat(1, 14, 1)
                disps_scale = G_model.synthesis(w_latent_repeat)
            else:
                disps_scale = G_model.synthesis(w_latent)
            disps = (disps_scale + 1) * 50.0
            loss = loss_function(disps, [left, right])
            loss_np = loss.cpu().detach().numpy()
            running_loss += loss_np
            loss.backward(retain_graph=True)
            optimizer.step()
            adjust_learning_rate(optimizer, step, args.learning_rate)

            if min_recons_loss > loss_np:
                min_recons_loss = loss_np
                cost_step = step+1
                disp_opt = disps.cpu().detach().numpy()
                w_opt = w_latent.cpu().detach().numpy()

        iter_end_time = time.time()
        G_total_times = (iter_end_time - iter_start_time)
        print("batch{} | recons_loss {} | cost step {}| P cost time{}|G cost time{}".
              format(batch + 1, min_recons_loss, cost_step, P_total_times, G_total_times))
        min_loss_list.append(min_recons_loss)
        cost_step_list.append(cost_step)
        w_out[batch * args.batch_size:(batch + 1) * args.batch_size] = w_opt
        disp_out[batch * args.batch_size:(batch + 1) * args.batch_size] = np.squeeze(disp_opt)

    avg_step = sum(cost_step_list) / len(cost_step_list)
    print(
        " loss_recon_avg {:.5f}".format(sum(min_loss_list) / len(min_loss_list)),
        " cost_step_avg {}".format(avg_step),
        " Total cost_time {}".format(P_total_times+G_total_times),
    )
    # 保存结果
    # np.save("final_result/disp_{}_test_p.npy".format(args.model), disp_out)
    # res_loss = np.array(min_loss_list)
    # np.save("final_result/loss_{}_test_p.npy".format(args.model), res_loss)

# ----------------------------------------------------------------------------

if __name__ == "__main__":

    """Train w using faster pmg model at our datasets

        Examples:

        \b
        # Train pmg for invivo.
        python stylegan3/PMGNet_train.py --dataset=invivo --model=pmg --mode=train

        \b
        # Test pmg model for invivo.
        python stylegan3/GNet_train.py --dataset=invivo --mode=pmg --mode=train_w

        or set default params below.
        如果报错 No module named 'utils'，在终端执行
        export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PMG-Net/aa_nets:/home/ubuntu/PMG-Net/stylegan3/dnnlib:/home/ubuntu/PMG-Net/stylegan3/torch_utils
        添加aanet和stylegan3所需的依赖库到环境变量中
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str,
                        help='[invivo, phantom]')
    parser.add_argument('--mode', default=None, type=str,
                        help='[train, test]')
    parser.add_argument('--model', default='mapping', type=str,
                        help='pmg' or 'mapping' or 'mg' or 'mean' or 'last')
    parser.add_argument('--data_output', default='final_models', type=str, help='Result')
    parser.add_argument('--data_dir', default=None, type=str, help='Training dataset')
    parser.add_argument('--gt_dir', default=None, type=str, help='Training dataset gt')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers for data loading')
    parser.add_argument('--learning_rate', default=5e-2, type=float, help='Learning rate')
    parser.add_argument('--upper', default=16, type=int, help=' upper cropped coodinate, invivo:32, phantom:16')
    parser.add_argument('--left', default=70, type=int, help='left cropped coodinate, invivo:14, phantom:70')
    parser.add_argument('--comp', default=-2.0, type=float, help='light compensate, invivo:4.3, phantom:-2.0')
    parser.add_argument('--max_epoch', default=20, type=int, help='Maximum epoch number for training')
    parser.add_argument('--max_step', default=150, type=int, help='Maximum iter number for training')
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
    args.gt_dir = 'datasets/{}_disp_gt.npy'.format(args.dataset)
    assert os.path.exists(args.gt_dir)
    pretrained_styleGAN = 'final_models/{}/optimizer_latest_sgan3.pkl'.format(args.dataset)
    assert os.path.exists(pretrained_styleGAN)
    print('=> Loading pretrained StyleGAN:', pretrained_styleGAN)
    with open(pretrained_styleGAN, 'rb') as f:
        G = legacy.load_network_pkl(f)['G'].to(args.device)
    if args.model == "pmg" or args.model == "mg":
        model = PMG(G, args).to(args.device)  # styleGAN's G
    elif args.model == "mapping":
        model = SiamG(G, args).to(args.device)
    args.data_dir = "datasets/{}/{}/".format(args.dataset, args.mode)
    print("=> Loading dataset from", args.data_dir)
    assert os.path.exists(args.data_dir)
    if args.mode == 'train':
        assert args.model in ['pmg', 'mg', 'mapping']
        train(args, model)
    else:
        assert args.model in ['pmg', 'mapping', 'mg', 'mean', 'last']
        test(args, G)

    # ----------------------------------------------------------------------------
# Record Invivo lr=5e-2 ms=150
# mapping(vgg+mapping): loss_recon_avg 62.48380  cost_step_avg 15.38
# pmg(aanet+mapping): loss_recon_avg 62.61963  cost_step_avg 12.34
# z域原点映射为初值(Zero): loss_recon_avg 63.44433  cost_step_avg 20.73
# 训练集最优w的均值为初值(w_train_opt-mean): loss_recon_avg 63.50888  cost_step_avg 23.35
# Record Phantom lr=5e-2 ms=150
# mapping(vgg+mapping): loss_recon_avg 42.29782  cost_step_avg 18.89
# pmg(aanet+mapping): loss_recon_avg 41.78096  cost_step_avg 15.75
# z域原点映射为初值(Zero): loss_recon_avg 41.55592 cost_step_avg 31.68 cost time 15458.8985s
# 训练集最优w的均值为初值(w_train_opt-mean): loss_recon_avg 41.47058 cost_step_avg 25.27

