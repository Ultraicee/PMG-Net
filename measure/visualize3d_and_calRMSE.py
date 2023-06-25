
import argparse
import os
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import numpy as np
import cv2


def read_an_image(img_name):
    img = cv2.imread(img_name, 1).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_stereo_images(source_img_path, id_list):
    # opencv读入的图像是BGR的格式
    left_path = os.path.join(source_img_path, r'image_2')
    right_path = os.path.join(source_img_path, r'image_3')
    left_list = os.listdir(left_path)
    right_list = os.listdir(right_path)
    left_list.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    right_list.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))

    if len(right_list) == len(left_list):
        left_name = os.path.join(left_path, left_list[id_list[0]])
        left_img = read_an_image(left_name)
        batch = len(id_list)
        height = np.shape(left_img)[0]
        width = np.shape(left_img)[1]
        channel = np.shape(left_img)[2]
        left_img_list = np.zeros([batch, height, width, channel], dtype=np.float32)
        right_img_list = np.zeros([batch, height, width, channel], dtype=np.float32)

        for i in range(0, len(id_list)):
            left_name = os.path.join(left_path, str(left_list[id_list[i]]))
            left_img_list[i, :, :, :] = read_an_image(left_name)
            right_name = os.path.join(right_path, str(right_list[id_list[i]]))
            right_img_list[i, :, :, :] = read_an_image(right_name)
    else:
        print('check your file')
    return left_img_list, right_img_list


def cal_xyz(disps,
            dataset,
            x0=0,
            y0=0):
    if dataset[-6:] == 'invivo':
        b = 5.49238
        f = 649.78255154
        u0 = 143.00123024
        v0 = 155.0463047
    else:
        b = 5.520739
        f = 445.72452766
        u0 = 173.584127434
        v0 = 149.79558372
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
    x0 = 54
    y0 = 42 if args.dataset == 'invivo' else 73
    ids = [x for x in range(100)]
    left_ims, right_ims = read_stereo_images(args.src_img_path, ids)  # shape=(100 256 256 3)
    right_clip = right_ims[:, x0:(x0 + args.img_size), y0:(y0 + args.img_size), :]
    point_geometry = [i for i in range(40, 200, 40)]  # 右图加黄线
    if args.dataset == 'invivo':
        x_offset = 54 - 32
        y_offset = 42 - 14
    else:
        x_offset = 54 - 16
        y_offset = 73 - 70
    for frame_idx in frame_idxs:
        # 绘制纹理线条
        for l in point_geometry:
            right_clip[frame_idx, l, :] = [255.0, 255.0, 0.0]
            right_clip[frame_idx, :, l] = [255.0, 255.0, 0.0]

        disp = disps[frame_idx, x_offset:(x_offset + args.img_size), y_offset:(y_offset + args.img_size)]
        world_position_i = right_clip[frame_idx] / 255
        world_position_x, world_position_y, world_position_z = cal_xyz(disp, args.dataset, x0=x0, y0=y0)
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
        if args.dataset == 'invivo':
            ax.set_xlim(-10, 7.5)
            ax.set_ylim(-10, 8)
            ax.set_zlim(-50, -38)
            ax.view_init(elev=40., azim=-20.)

            ax.set_xlabel('Y(mm)')
            ax.set_ylabel('X(mm)')
            ax.set_zlabel('Z(mm)')

            ax.set_xticks([i for i in range(-10, 10, 4)])
            ax.set_yticks([i for i in range(-10, 10, 4)])
            ax.set_zticks([-50, -46, -42, -38])
            ax.set_zticklabels(['50', '46', '42', '38'])
        else:
            ax.set_xlim(-15, 20)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-65, -40)
            ax.view_init(elev=40., azim=-20.)

            ax.set_xlabel('Y(mm)')
            ax.set_ylabel('X(mm)')
            ax.set_zlabel('Z(mm)')

            ax.set_zticks([-65, -60, -55, -50, -45, -40])
            ax.set_zticklabels(['65', '60', '55', '50', '45', '40'])

        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        # 修改存储路径
        if args.save_path:
            # plt.savefig(args.save_path + '/{}_3D_{}_'.format(args.dataset, args.model) + str(frame_idx + 1) + '.png', dpi=200,
            #             bbox_inches='tight', pad_inches=0)
            plt.savefig('../final_result/robust_exp/{}_{}_obs_{}'.format(args.dataset, args.model, frame_idx*10) + '.png',
                        dpi=200,
                        bbox_inches='tight', pad_inches=0)
        plt.show()
        print('../final_result/robust_exp/{}_{}_obs_{}'.format(args.dataset, args.model, frame_idx*10) + '.png')
        plt.close()


def cal_rmse(predict_disps, loc_disp_gt, args):
    """
    基于显著特征点计算匹配误差
    Args:
        predict_disps: 模型预测的视差值
        loc_disp_gt: 特征子计算的特视差值
        args:

    Returns: RMSE

    """
    if args.dataset == 'invivo':
        x_offset = 54 - 32
        y_offset = 42 - 14
    else:
        x_offset = 54 - 16
        y_offset = 73 - 70
    loc_xy_200 = (loc_disp_gt[:, :, 0:-1]+0.5).astype(int)  # 就近邻
    disp_gt = loc_disp_gt[:, :, -1]
    predict_disps_sample = np.zeros((100, 5))
    valid_num = 0
    for i in range(100):
        for j in range(5):
            # 200x200的ROI区域坐标系->256x256的视差图坐标系
            x_256, y_256 = loc_xy_200[i][j][0] + x_offset, loc_xy_200[i][j][1] + y_offset
            if disp_gt[i][j] > 0.0:  # 筛选有效视差数据
                valid_num += 1
                predict_disps_sample[i][j] = predict_disps[i][x_256, y_256]
    mse = mean_squared_error(predict_disps_sample, disp_gt) * (500 / valid_num)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3d reconstruction for our dataset')
    parser.add_argument('--img_size', type=int, default=200,
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
        model_list = ['sgan2', 'diffusion_sgan2', 'sgan3', 'aanet', 'vae']
    else:
        model_list = [args.model]
    assert args.dataset in ['invivo', 'phantom']
    print("test {} dataset".format(args.dataset))
    # shape=(100,5,3), 基于ORB算子+暴力匹配得到的准确匹配视差, 具体见feature_match.py
    loc_disp_gt = np.load("../final_result/{}_feature_match_result/R_img_sz200_pts_and_disps.npy".format(args.dataset))
    predict_disps = np.load("../final_result/robust_exp/disp_obs_test_frame615_invivo_all.npy")
    for i, model in enumerate(model_list):
        args.model = model
        """ 计算显著性匹配误差 """
        predict_disps = np.load("../final_result/{}/disp_{}_test_{}.npy".format(args.dataset, model, args.dataset[0]))
        print("Calculate {} disparity loss......".format(model))
        rmse = cal_rmse(predict_disps, loc_disp_gt, args)
        print("{} | {:.4f}".format(model, rmse))

        # """ 生成3D图像 """
        # print("plot {} selected frame 3D figure......".format(model))
        # reconstruction(args, predict_disps[i], frame_idxs=[x for x in range(6)])
        # print("plot finished!")

# Model      | Invivo   | Phantom
# tps16      | 1.9180   | 2.0789
# tps25      | 2.0786   | 2.3526
# sgan       | 2.1146   | 2.2238
# sgan2      | 2.1657   | 2.3799
# dfs-sgan2  | 2.1526   | 2.4703
# sgan3      | 1.9266   | 2.2301
# aanet      | 2.2686   | 2.4637
# vae        | 2.6174   | 2.3603
