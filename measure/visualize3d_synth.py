import argparse
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import numpy as np
import cv2


def read_an_image(img_name):
    img = cv2.imread(img_name, 1).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cal_xyz(disps,
            x0=0,
            y0=0):

    b = 5.49238
    f = 649.78255154
    u0 = 143.00123024
    v0 = 155.0463047
    img_size = disps.shape[-2:]
    x = np.array([i for i in range(1, 1 + img_size[0])])
    y = np.array([i for i in range(1, 1 + img_size[1])])
    X, Y = np.meshgrid(x, y)

    X = X + y0
    Y = Y + x0

    out_X = b * (X - u0) / disps
    out_Y = b * (Y - v0) / disps
    out_Z = b * f / disps
    return out_X, out_Y, out_Z


def reconstruction(args, disps, frame_idxs=[100, 500, 1000]):
    # 跟ROI区域选择有关
    x0 = 42
    y0 = 37

    point_geometry = [i for i in range(40, args.img_size, 40)]  # 右图加黄线

    for n, frame_idx in enumerate(frame_idxs):
        right_im = read_an_image("/home/ubuntu/WS-YG/PMG-Net/datasets/synth_invivo/test/image_3/right_{}.jpg".format(frame_idx+1))
        right_clip = right_im[x0:(x0 + args.img_size), y0:(y0 + args.img_size), :]
        # 绘制纹理线条
        for l in point_geometry:
            right_clip[l, :] = [255.0, 255.0, 0.0]
            right_clip[:, l] = [255.0, 255.0, 0.0]

        disp = disps[frame_idx]
        world_position_i = right_clip / 255
        world_position_x, world_position_y, world_position_z = cal_xyz(disp, x0=x0, y0=y0)
        world_position_x = world_position_x.reshape([1, -1])
        world_position_y = world_position_y.reshape([1, -1])
        world_position_z = -1 * world_position_z.reshape([1, -1])
        world_position_i = world_position_i.reshape([1, -1, 3])
        # 四角处理
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter3D(world_position_y[0], world_position_x[0], world_position_z[0], c=world_position_i[0], s=0.1,
                     norm=1, alpha=1)

        # ax.set_xlim(-10, 7.5)
        # ax.set_ylim(-10, 8)
        # ax.set_zlim(-50, -38)
        # ax.view_init(elev=40., azim=-20.)

        ax.set_xlabel('Y(mm)')
        ax.set_ylabel('X(mm)')
        ax.set_zlabel('Z(mm)')

        # ax.set_xticks([i for i in range(-10, 10, 4)])
        # ax.set_yticks([i for i in range(-10, 10, 4)])
        # ax.set_zticks([-50, -46, -42, -38])
        # ax.set_zticklabels(['50', '46', '42', '38'])
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        # 修改存储路径
        if args.save_path:
            plt.savefig('../final_result/{}_{}_{}_3d_result'.format(args.dataset, args.model, frame_idx+1) + '.png',
                    dpi=200,
                    bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3d reconstruction for our dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='reconstruction valid image size')
    parser.add_argument('--save_path', type=str, default='3d_result/',
                        help='save reconstruction result, check the path existed.')
    parser.add_argument('--dataset', type=str, default='invivo',
                        help='dataset name')
    parser.add_argument('--model', type=str, default=None,
                        help='model name')
    parser.add_argument('--src_img_path', type=str, default=None,
                        help='load dataset')
    args = parser.parse_args()
    args.src_img_path = "../datasets/{}/test".format(args.dataset)
    if args.model is None:
        # model_list = ['tps16', 'tps25', 'sgan', 'sgan2', 'diffusion_sgan2', 'sgan3', 'aanet', 'vae']
        model_list = ['sgan2']
    else:
        model_list = [args.model]
    assert args.dataset in ['invivo', 'phantom', 'synth_invivo']
    print("test {} dataset".format(args.dataset))
    # shape=(100,5,3), 基于ORB算子+暴力匹配得到的准确匹配视差, 具体见feature_match.py
    # loc_disp_gt = np.load("../final_result/{}_feature_match_result/R_img_sz200_pts_and_disps.npy".format(args.dataset))

    for i, model in enumerate(model_list):
        args.model = model
        predict_disps = np.load("/home/ubuntu/WS-YG/PMG-Net/datasets/synth_invivo/test/synth_invivo_disp4test.npy")
        predict_disps = predict_disps[:, 42:42+128, 37:37+128]
        print("plot {} selected frame 3D figure......".format(model))
        reconstruction(args, predict_disps)
        print("plot finished!")
