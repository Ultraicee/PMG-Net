"""Measure final disps file‘s Similarity Measure Index of testing datasets."""
import os
import numpy as np
import argparse
import torch
from utils import to_device
from dataloader.dataloader import prepare_dataloader
from utils.loss import DepthNetLoss
import warnings

warnings.filterwarnings("ignore")


def measure(args):
    fname = '../final_result/{}/disp_{}_test_{}.npy'.format(args.dataset, args.model, args.dataset[0])
    assert os.path.exists(fname)
    disps = np.load(fname)

    n_img, loader = prepare_dataloader(args.data_dir, None, 1, 3, False)
    depth_loss = DepthNetLoss(SSIM_w=0.0, disp_gradient_w=0.0, img_w=0.0, left=args.left, upper=args.upper,
                              compensateI=args.comp).to(args.device)
    mse_all = 0.0
    ssim_all = 0.0
    for batch, data in enumerate(loader):
        data = to_device(data, 'cuda:0')
        left = data['left_image']
        right = data['right_image']
        target = torch.tensor(disps[batch, :, :], dtype=torch.float32, device='cuda:0', requires_grad=False)
        depth_loss(target, [left, right])
        mse_all += depth_loss.loss_pho.cpu().numpy()
        ssim_all += depth_loss.ssim_valid.cpu().numpy()
    mse_avg = mse_all / n_img
    ssim_avg = ssim_all / n_img
    psnr_avg = 10 * np.log10(255 * 255 / mse_avg)
    print(args.model, ": mse={:3f}, psnr={:3f}, ssim={:3f}".format(mse_avg, psnr_avg, ssim_avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tps25', type=str,
                        help='model name')
    parser.add_argument('--upper', default=16, type=int, help=' upper cropped coodinate, invivo:32, phantom:16')
    parser.add_argument('--left', default=70, type=int, help='left cropped coodinate, invivo:14, phantom:70')
    parser.add_argument('--comp', default=-2.0, type=float, help='light compensate, invivo:4.3, phantom:-2.0')
    parser.add_argument('--dataset', default='phantom', type=str, help='measure dataset')
    parser.add_argument('--data_dir', default=None, type=str, help='dataset path')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')

    args = parser.parse_args()
    if args.dataset == 'invivo':
        args.upper = 32
        args.left = 14
        args.comp = 4.3
    args.data_dir = '../datasets/{}/test'.format(args.dataset)
    if args.model is None:
        model_list = ['tps16', 'tps25', 'sgan', 'sgan2', 'sgan3', 'diffusion_sgan2', 'aanet', 'vae']
    else:
        model_list = [args.model]

    for model in model_list:
        args.model = model
        measure(args)
    print("Done!")

# Record, Invivo dataset
# tps16 : mse=63.779157, psnr=30.084016, ssim=0.910560
# tps25 : mse=62.008212, psnr=30.206312, ssim=0.913398
# sgan : mse=63.192827, psnr=30.124126, ssim=0.911185
# sgan2 : mse=64.844152, psnr=30.012095, ssim=0.910716
# sgan3 : mse=62.901010, psnr=30.144227, ssim=0.913264
# d-sgan2 : mse=64.520044, psnr=30.033857, ssim=0.910196
# aanet : mse=65.138764, psnr=29.992408, ssim=0.907922
# vae : mse=70.274062, psnr=29.662853, ssim=0.898438

# ---------------------------------------------------------

# Record, Phantom dataset
# tps16 : mse=43.511685, psnr=31.744745, ssim=0.899755
# tps25 : mse=41.942183, psnr=31.904293, ssim=0.902985
# sgan : mse=41.317970, psnr=31.969414, ssim=0.906142
# sgan2 : mse=41.518480, psnr=31.948389, ssim=0.904131
# sgan3 : mse=41.127423, psnr=31.989489, ssim=0.904982
# d-sgan2 : mse=41.481292, psnr=31.952281, ssim=0.904398
# aanet : mse=41.671513, psnr=31.932411, ssim=0.904128
# vae : mse=43.408956, psnr=31.755010, ssim=0.900628
#
