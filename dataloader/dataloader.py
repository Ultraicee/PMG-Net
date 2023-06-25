import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class StereoDataset(Dataset):
    """
    双目内窥镜数据集
    """
    def __init__(self, root_dir, gt_dir=None, transform=None):
        left_dir = os.path.join(root_dir, 'image_2/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname \
                                  in os.listdir(left_dir)])
        right_dir = os.path.join(root_dir, 'image_3/')
        self.right_paths = sorted([os.path.join(right_dir, fname) for fname \
                                   in os.listdir(right_dir)])
        assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.gt = None
        if gt_dir:
            print("=> Use gt data from", gt_dir)
            self.gt = np.load(gt_dir)

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = imread(self.left_paths[idx])
        right_image = imread(self.right_paths[idx])

        # 默认执行中值滤波，处理噪声
        left_image = cv2.medianBlur(left_image, 3)
        right_image = cv2.medianBlur(right_image, 3)

        # 鲁棒性实验进行数据处理的接口
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        left_image_np = np.array(left_image).astype(np.float32)
        right_image_np = np.array(right_image).astype(np.float32)
        # HWC->CHW
        left_image_np = np.transpose(left_image_np, (2, 0, 1))
        right_image_np = np.transpose(right_image_np, (2, 0, 1))

        if self.gt is not None:
            gt_np = self.gt[idx]
            sample = {'left_image': left_image_np, 'right_image': right_image_np, 'gt': gt_np}
        else:
            sample = {'left_image': left_image_np, 'right_image': right_image_np}
        return sample


def get_robust_transforms(obstacle_sz=0, blur_kernel_sz=0):
    """
    执行鲁棒性实验所需的数据变换
    Args:
        obstacle_sz: 遮挡大小
        blur_kernel_sz: 模糊核大小，越大模糊程度越高

    Returns:

    """
    trans_list = []
    if obstacle_sz > 0:
        trans_list.append(
            A.CoarseDropout(max_holes=1, max_height=obstacle_sz, max_width=obstacle_sz,
                            min_holes=1, min_height=obstacle_sz, min_width=obstacle_sz,
                            fill_value=0, always_apply=True, p=1.0))
    if blur_kernel_sz > 0:
        trans_list.append(A.MotionBlur(blur_limit=blur_kernel_sz, always_apply=True, p=1.0))
    return A.Compose(
        trans_list,
        p=1.0,
    )


def prepare_dataloader(data_directory, gt_dir, batch_size, num_workers, Shuffle_Flag=True):
    """
    准备数据集加载器
    Args:
        data_directory: 图像数据集所在目录
        gt_dir: 标签所在目录
        batch_size: 批大小
        num_workers: 工作线程数
        Shuffle_Flag: True时开启数据随机操作

    Returns:
        n_img: 数据集大小
        loader: 数据迭代器
    """
    dataset = StereoDataset(os.path.join(data_directory), gt_dir, transform=None)
    n_img = len(dataset)
    print('=> Use a dataset with', n_img, 'images')
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=Shuffle_Flag, num_workers=num_workers,
                        pin_memory=True)

    return n_img, loader


def imread(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 功能：函数cvCvtColor实现色彩空间转换
    image = image.astype(np.uint8)
    return np.array(image)


def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 读取图片及变换测试
    image_l = r"..\datasets\invivo\test\image_2\rect_left_0615.bmp"
    image_r = r"..\datasets\invivo\test\image_2\rect_left_0615.bmp"
    a = imread(image_l)
    b = imread(image_r)
    for bk_sz in range(3, 25, 2):
        trans = get_robust_transforms(obstacle_sz=0, blur_kernel_sz=bk_sz)
        # print("obstacle size:", obs_sz)
        print("blur kernel size:", bk_sz)
        image2 = trans(image=a)['image']
        image3 = trans(image=b)['image']
        show(image2)
        # 查看鲁棒性测试图片效果
        plt.imsave(r"..\datasets\invivo\robust_test\rect_left_0615_bk_sz={}.bmp".format(bk_sz), image2)
        plt.imsave(r"..\datasets\invivo\robust_test\rect_right_0615_bk_sz={}.bmp".format(bk_sz), image3)
