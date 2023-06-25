import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def cal_xyz(disps,
            dataset_name,
            x0=0,
            y0=0):
    """
    计算视差图对应的空间坐标数据
    Args:
        disps: 视差图
        dataset_name: 数据集名称
        x0:
        y0:

    Returns:

    """
    if dataset_name == 'invivo':
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


def cal_disp(depth, dataset_name='invivo'):
    if dataset_name == 'invivo':
        b = 5.49238
        f = 649.78255154
    else:
        b = 5.520739
        f = 445.72452766
    return b * f / depth


def video2disp(video_path, output_dir, video_idx):
    """
    采样RGB-D视频数据并保存为图片格式
    Args:
        video_path:
        output_dir:
        video_idx:

    Returns:

    """
    times = 0
    blank = 10
    if not os.path.exists(output_dir):
        # 如果文件目录不存在则创建目录
        os.makedirs(output_dir)
    camera = cv2.VideoCapture(video_path)
    frame_num = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_height = int((frame_height - blank) / 2)
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratio = 2
    disp_data = np.zeros((frame_num // ratio, frame_height, frame_width))
    idx = 0
    while True:
        times += 1
        res, image = camera.read()
        if not res or times > frame_num:
            break

        if times % ratio == 0:  # 低频采样
            depth_data = 0.114 * image[frame_height + blank:, :, 0] + \
                         0.587 * image[frame_height + blank:, :, 1] + \
                         0.299 * image[frame_height + blank:, :, 1]
            disp_data[idx] = cal_disp(depth_data, dataset_name='invivo')
            img = torch.from_numpy(image[:frame_height]).unsqueeze(-1).permute(3, 2, 0, 1).float()  # [NCHW]
            disp = torch.from_numpy(disp_data[idx]).unsqueeze(0)  # [NHW]
            left_recons = apply_disparity(img, disp)

            # 保存数据
            left = torch.squeeze(left_recons).permute(1, 2, 0).numpy()
            left = left.astype(np.uint8)
            idx += 1
            cv2.imwrite(os.path.join(output_dir, r"image_2/left_" + str(idx + video_idx * 512) + '.jpg'), left)
            print(os.path.join(output_dir, r"image_2/left_" + str(idx + video_idx * 512) + '.jpg'))
            cv2.imwrite(os.path.join(output_dir, r"image_3/right_" + str(idx + video_idx * 512) + '.jpg'),
                        image[:frame_height])
            print(os.path.join(output_dir, r"image_3/right_" + str(idx + video_idx * 512) + '.jpg'))

    print("disp.shape=", disp_data.shape)
    print("disp min=", np.min(disp_data), "and max=", np.max(disp_data))
    assert np.min(disp_data) > 0
    print('{}提取结束，共采样{}帧\n'.format(video_path, idx))
    camera.release()
    return disp_data


def apply_disparity(img, disp):
    """
    基于任一视角的RGB图像与对应视差图生成另一视角图像
    Args:
        img:
        disp:

    Returns:

    """
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                 width, 1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp / width  # Disparity is passed in NHW format
    x_shifts = x_shifts.squeeze(1)

    flow_field = torch.stack((x_base - x_shifts, y_base), dim=3).type_as(img)  # R(x,y-d)=L‘
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                           padding_mode='zeros')
    return output


if __name__ == "__main__":
    output_dir = r"D:\Pytorch\PMG-Net\datasets\syth_invivo\test"
    videos_path = r"D:\Pytorch\PMG-Net\datasets\syth_invivo\videos"
    videos_name_list = os.listdir(videos_path)
    sorted(videos_name_list)

    # # train
    # part_num = 12 # [12 4 4]=[train valid test]
    # part_videos_name_list = videos_name_list[:part_num]

    # test
    part_num = 4  # [12 4 4]=[train valid test]
    offset = 12
    part_videos_name_list = videos_name_list[12:12 + part_num]
    disps_data = np.zeros((512 * part_num, 256, 256))

    for video_name in part_videos_name_list:
        print("video name:", video_name)
        video_path = os.path.join(videos_path, video_name)
        video_idx = int(video_name[-10:-4]) - offset
        disps_data[video_idx * 512:(video_idx + 1) * 512] = video2disp(video_path, output_dir, video_idx)
    np.save(r"..\datasets\syth_invivo\test\Synth_invivo_disp4test.npy", disps_data)
    print("Finished, final disp min=", np.min(disps_data), "and max=", np.max(disps_data))
