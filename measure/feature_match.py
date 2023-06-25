"""Tool for creating distinctive feature points."""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def feature_match(args):
    # opencv读入的图像是BGR的格式
    left_path = os.path.join(args.src_path, r'image_2')
    right_path = os.path.join(args.src_path, r'image_3')
    left_list = os.listdir(left_path)

    right_list = os.listdir(right_path)
    left_list.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    right_list.sort(key=lambda x: int(x.split(".")[0].split("_")[2]))
    orb = cv2.ORB_create()
    # 初始化 BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    img_num = len(left_list)
    conf_gt = np.zeros((img_num, 5, 3))
    x0 = 54
    # y0 = 42  # invivo
    y0 = 73  # phantom
    roi_size = 200
    for i in range(img_num):
        # 寻找关键点
        left = cv2.imread(os.path.join(left_path, left_list[i]), 1)
        # left = cv2.medianBlur(left, 3)
        left = left[x0:x0 + roi_size, y0:y0 + roi_size, :]
        right = cv2.imread(os.path.join(right_path, right_list[i]), 1)
        # right = cv2.medianBlur(right, 3)
        right = right[x0:x0 + roi_size, y0:y0 + roi_size, :]
        detected_num = 0
        while detected_num < 10:
            kp1 = orb.detect(left)
            kp2 = orb.detect(right)
            # 计算描述符
            kp1, des1 = orb.compute(left, kp1)
            kp2, des2 = orb.compute(right, kp2)
            # 对描述子进行匹配

            matches = bf.match(des1, des2)
            detected_num = len(matches)
        matches = sorted(matches, key=lambda x: x.distance)
        print("frame{:3d} min_dist={:.3f}, max_dist={:.3f}".format(i + 1, matches[0].distance, matches[-1].distance))
        good_match = []
        x_last, y_last = 0, 0
        for x in matches:
            if len(good_match) < 5:
                x1 = kp1[x.queryIdx].pt[0]
                y1 = kp1[x.queryIdx].pt[1]
                x2 = kp2[x.trainIdx].pt[0]
                y2 = kp2[x.trainIdx].pt[1]
                if abs(y1 - y2) > 1 or (x1 - x2) < 0 or abs(x2 - x_last) + abs(y1 - y_last) < 5:  # y轴平行且不邻近
                    continue
                else:
                    disp = x1 - x2
                    x_last, y_last = x2, y2  # 去掉邻近点
                    conf_gt[i, len(good_match)] = np.array([x2, y2, disp])
                    good_match.append(x)
        outimage = cv2.drawMatches(left, kp1, right, kp2, good_match, outImg=None, matchesThickness=1)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(args.save_path, "frame{}.jpg".format(i + 1)), outimage)
        np.save(os.path.join(args.save_path, "R_img_sz200_pts_and_disps.npy"), conf_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='feature match for our dataset')
    parser.add_argument('--img_size', type=int, default=200,
                        help='reconstruction valid image size')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save reconstruction result, check the path existed.')
    parser.add_argument('--dataset', type=str, default='phantom',
                        help='dataset name')
    parser.add_argument('--src_path', type=str, default=None,
                        help='load dataset')
    args = parser.parse_args()
    args.save_path = '../final_result/{}_feature_match_result/'.format(args.dataset)
    args.src_path = "../datasets/{}/test".format(args.dataset)
    feature_match(args)  # 生成匹配真值
