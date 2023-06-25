import numpy as np
import imageio

img_size = 256
disps = np.load('D:\Pytorch\PMG-Net\datasets\invivo_disp_gt.npy')

model_list = ['tps16', 'tps25', 'sgan3', 'sgan2', 'diffusion_sgan2', 'vae', 'aanet']
idx_list = [24, 49, 74]
for model in model_list:

    imgs_test = np.load('D:/Pytorch/PMG-Net/final_result/invivo/disp_{}_test_i.npy'.format(model))
    imgs_test = imgs_test.reshape((len(imgs_test), 256, 256, 1))
    print("model:", model, imgs_test.shape)

    # 进行拉伸
    maxnum = imgs_test.max()
    minnum = imgs_test.min()
    maxnum = max(maxnum, disps.max())
    minnum = min(minnum, disps.min())
    scale = maxnum - minnum
    imgs_test = (imgs_test - minnum) / scale * 255
    imgs_test -= 80
    for i in idx_list:
        B = imgs_test[i]
        imageio.imsave("./disp figure/disp" + str(i + 1) + "_{}_i.png".format(model), B.astype(np.uint8))
