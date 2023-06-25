import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False,
}
rcParams.update(config)


def plot_loss_curve(model_list, dataset="invivo"):
    x = [i for i in range(1, 101)]
    plot_style = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    fig, ax = plt.subplots()  # 创建图实例
    for model in (model_list):
        loss = np.load("../final_result/{}/loss_{}_test_{}.npy".format(dataset, model, dataset[0]))
        ax.plot(x, loss, label=model, linestyle=plot_style[0])
    ax.set_title('SIMILARITY_MEASURES({})'.format(dataset))
    ax.set_xlabel('Img', fontsize=15)  # 设置x轴名称 x label
    # ax.set_ylabel('MSE') #设置y轴名称 y label
    ax.set_ylabel('$loss_{pho}$', fontsize=15)  # 设置y轴名称 y label

    ax.legend()  # 自动检测要在图例中显示的元素，并显示
    plt.xlim(1, loss.shape[0])
    # plt.ylim(50, 100)
    # plt.savefig('../final_result/{}_SIMILARITY_MEASURES_test_rec_loss.tif'.format(dataset), dpi=600, facecolor='w',
    #             edgecolor='w', transparent=False)
    plt.show()
    print("Done!")


def plot_training_loss_curve():
    # valid loss each epoch
    mg_train_loss = [0.01640, 0.01491, 0.01484, 0.01253, 0.01263, 0.01063, 0.01149, 0.01097, 0.01025, 0.01077,
                     0.01017, 0.01032, 0.01207, 0.01134, 0.00999, 0.01057, 0.01054, 0.01041, 0.01019, 0.00976]
    mapping_train_loss = [0.11969, 0.07407, 0.02288, 0.02068, 0.01706, 0.01783, 0.01639, 0.01776, 0.01540, 0.01625,
                          0.01543, 0.01404, 0.01436, 0.01309, 0.01231, 0.01237, 0.01190, 0.01186, 0.01179, 0.01331]
    pmg_train_loss = [0.03202, 0.02037, 0.01488, 0.01268, 0.01457, 0.01027, 0.01049, 0.00975, 0.00925, 0.00877,
                      0.00817, 0.00832, 0.00961, 0.01017, 0.00938, 0.00876, 0.00926, 0.00898, 0.00819, 0.00884]

    w_opt = np.load("../datasets/w_latent600_sgan3_phantom.npy")  # [600,1,16]
    w_mean_pre = np.mean(w_opt[:500, 0, :], axis=0)
    w_mean_pre = np.expand_dims(w_mean_pre, 0).repeat(99, axis=0)
    mean_pre_loss = mean_squared_error(w_mean_pre, w_opt[500:599, 0, :])
    last_pre_loss = mean_squared_error(w_opt[501:600, 0, :], w_opt[500:599, 0, :])
    w_shape = w_opt.shape
    plot_style = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    fig, ax = plt.subplots()  # 创建图实例
    x = [i for i in range(1, 21)]
    ax.plot(x, mg_train_loss, label="MG-Net", linestyle=plot_style[0])
    ax.plot(x, pmg_train_loss, label="PMG-Net", linestyle=plot_style[0])
    ax.plot(x, mapping_train_loss, label="SiamG-Net", linestyle=plot_style[0])
    ax.plot(x, [last_pre_loss] * 20, label="Last predict", linestyle=plot_style[0])
    ax.plot(x, [mean_pre_loss] * 20, label="Mean predict", linestyle=plot_style[0])
    # for dim in [1, 2, 3]:
    #     ax.plot(x, w_opt[400:500, 0, dim], label="dim" + str(dim), linestyle=plot_style[0])
    # ax.set_xlabel('epoch', fontsize=15)  # 设置x轴名称 x label

    ax.set_xlabel('训练轮次($epoch$)', fontsize=12)  # 设置x轴名称 x label
    ax.set_ylabel('$MSE(mm^2)$', fontsize=12)  # 设置y轴名称 y label
    plt.subplots_adjust(bottom=0.15, left=0.15)
    ax.legend()  # 自动检测要在图例中显示的元素，并显示
    plt.savefig("TrainingLossEachEpoch.jpg", dpi=600)
    plt.show()


def plot_SampleSingleDimensionData():
    w_opt = np.load("../datasets/w_latent600_sgan3_phantom.npy")  # [600,1,16]
    plot_style = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    fig, ax = plt.subplots()  # 创建图实例
    start_pt = 460
    length = 100
    x = [i for i in range(start_pt, start_pt + length)]
    for dim in [1, 2, 3]:
        ax.plot(x, w_opt[start_pt: start_pt + length, 0, dim], label="dim" + str(dim), linestyle=plot_style[0])
    ax.set_xlabel("帧", fontsize=15)  # 设置x轴名称 x label
    ax.set_ylabel("向量值", fontsize=15)  # 设置y轴名称 y label

    plt.subplots_adjust(bottom=0.15, left=0.15)
    ax.legend(loc='upper right')  # 自动检测要在图例中显示的元素，并显示
    plt.savefig("SampleSingleDimensionData.jpg", dpi=600)
    plt.show()


def concat_3d_figs(model_list, dataset="invivo"):
    Col = len(model_list)
    Rol = 3
    col_list = []
    idx_list = [25, 50, 75]

    for r in range(Rol):
        rol_list = []
        for c in range(Col):
            path = "../final_result/3d_result/{}_3D_{}_{}.png".format(dataset, model_list[c], idx_list[r])
            mat = cv2.imread(path)
            rol_list.append(mat)
        img = np.hstack(rol_list)  # 水平拼接
        col_list.append(img)
    imgs = np.vstack(col_list)  # 垂直拼接
    cv2.imwrite("../final_result/3d_result/{}_3d_recons_result2.png".format(dataset), imgs)


if __name__ == "__main__":
    # plot_training_loss_curve()
    plot_SampleSingleDimensionData()
