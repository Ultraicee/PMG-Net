"""train and test aanet network pickle."""
import os
import argparse
import time

import torch
from torch import optim, nn

import aa_nets
import numpy as np
from utils.loss import DepthNetLoss
from utils import to_device, prepare_dataloader, load_pretrained_net

import warnings

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
def train(args, model):
    n_img, loader = prepare_dataloader(args.data_dir, args.gt_dir, args.batch_size, args.num_workers)
    loss_function = DepthNetLoss(SSIM_w=0.0, disp_gradient_w=0.0, img_w=0.0, left=args.left, upper=args.upper,
                                 compensateI=args.comp).to(args.device)
    optimizer = optim.Adam([{"params": model.parameters()}],
                           lr=args.learning_rate)

    train_batch_num = n_img // args.batch_size

    model.train()
    min_ave = 10086.0
    args.batch_size = 5
    loss_fn = nn.MSELoss()
    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for batch, data in enumerate(loader):
            # Load data
            data = to_device(data, args.device)
            left = data['left_image']
            right = data['right_image']
            disps_gt = data['gt']

            disps = model(left, right)[-1]  # [B, H, W]
            disps = disps[:, args.upper:args.upper + 256, args.left:args.left + 256]
            loss = loss_fn(disps, disps_gt)  # labels MSE
            loss_function(disps.unsqueeze(1), [left, right])  # MonoDepth loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy()

        epoch_end_time = time.time()
        epoch_cost_time = epoch_end_time - epoch_start_time
        m = int(epoch_cost_time) // 60
        s = int(epoch_cost_time) % 60
        curr_ave = epoch_loss / train_batch_num
        print("Epoch[{}/{}]".format(epoch + 1, args.max_epoch),
              " average loss {:.5f}".format(curr_ave),
              " cost time {}m{}s".format(int(m), int(s))
              )
        if curr_ave < min_ave:
            min_ave = curr_ave
            # Save model
            PATH = "final_models/{}_optimizer_latest_aanet.pkl".format(args.dataset)  # 仅放在外部文件夹内
            print("=> Save model:%s" % PATH)
            torch.save(model.state_dict(), PATH)


# ----------------------------------------------------------------------------

def test(args, G_model):
    n_img, loader = prepare_dataloader(args.data_dir, args.gt_dir, args.batch_size, args.num_workers, False)
    min_loss_list = []
    args.batch_size = 1
    disp_out = np.zeros((n_img, 256, 256))
    loss_function = DepthNetLoss(SSIM_w=0.0, disp_gradient_w=0.0, img_w=0.0, left=args.left, upper=args.upper,
                                 compensateI=args.comp).to(args.device)
    for batch, data in enumerate(loader):
        # Load data
        data = to_device(data, args.device)
        left = data['left_image']
        right = data['right_image']

        disps = G_model(left, right)[-1]  # [B, H, W]
        disps = disps[:, args.upper:args.upper + 256, args.left:args.left + 256]
        loss = loss_function(disps.unsqueeze(1), [left, right])
        min_loss_list.append(loss.cpu().detach().numpy())
        disp_out[batch * args.batch_size:(batch + 1) * args.batch_size] = disps.cpu().detach().numpy()
        print("batch{} | recons_loss {}".format(batch + 1, loss.item()))

    res_loss = np.array(min_loss_list)
    np.save("result/loss_aanet_{}_{}.npy".format(args.mode, args.dataset), res_loss)
    np.save("result/disp_aanet_{}_{}.npy".format(args.mode, args.dataset), disp_out)


# ----------------------------------------------------------------------------


if __name__ == "__main__":

    """Train AA-Net model using our datasets

        Examples:

        \b
        # Train AA-Net for invivo.
        python aanet_train_test.py --dataset=invivo --mode=train

        \b
        # Test AA-Net for invivo.
        python aanet_train_test.py --dataset=invivo --mode=test

        or set default params below.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str,
                        help='[invivo, phantom]')
    parser.add_argument('--mode', default=None, type=str,
                        help='[train, test]')
    parser.add_argument('--data_dir', default='datasets/phantom/test', type=str, help='Training dataset')
    parser.add_argument('--gt_dir', default=None, type=str, help='Testing dataset')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers for data loading')
    parser.add_argument('--upper', default=16, type=int, help=' upper cropped coodinate, invivo:32, phantom:16')
    parser.add_argument('--left', default=70, type=int, help='left cropped coodinate, invivo:14, phantom:70')
    parser.add_argument('--comp', default=-2.0, type=float, help='light compensate, invivo:4.3, phantom:-2.0')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
    parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoch number for training')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    parser.add_argument('--pretrained_aanet', default=None, type=str,
                        help='Pretrained aanet')

    args = parser.parse_args()
    # 加载特征聚合模块
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

    assert args.dataset in ["invivo", "phantom"]
    if args.dataset == 'invivo':
        args.upper = 32
        args.left = 14
        args.comp = 4.3
    args.gt_dir = 'datasets/{}_disp_gt.npy'.format(args.dataset)
    assert os.path.exists(args.gt_dir)
    args.pretrained_aanet = 'final_models/{}/optimizer_latest_aanet.pkl'.format(args.dataset)
    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    args.data_dir = "datasets/{}/{}/".format(args.dataset, args.mode)
    assert os.path.exists(args.data_dir)
    if args.mode == 'test':
        assert os.path.exists(args.pretrained_aanet)
        test(args, aanet)
    elif args.mode == 'train':
        train(args, aanet)
    print("Done!")
# ----------------------------------------------------------------------------
