import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    plt.savefig('../final_result/{}_SIMILARITY_MEASURES_test_rec_loss.tif'.format(dataset), dpi=600, facecolor='w',
                edgecolor='w', transparent=False)
    plt.show()
    print("Done!")


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


def plot_robust_figs():
    model_list = ['sgan2', 'diffusion_sgan2', 'sgan3', 'vae', 'aanet']
    measure_list = ['motion_blur', 'obstacle']
    for measure in measure_list:
        fig, ax = plt.subplots()  # 创建图实例
        data = np.load("D:/yg_graduation/PMG-Net/final_result/robust_exp/loss_{}_test_frame615_invivo.npy".format(measure))

        if measure == 'motion_blur':
            ax.set_xlabel('degree of motion', fontsize=15)  # 设置x轴名称 x label
            ax.set_title('Motion blur test'.format(measure))
            x = np.arange(data.shape[1])
        else:
            ax.set_xlabel('size of obstacle', fontsize=15)  # 设置x轴名称 x label
            ax.set_title('Obstacle test')
            x = np.arange(0, data.shape[1]*10, 10)

        for i, model in enumerate(model_list):
            ax.plot(x, data[i], label=model, linestyle='-')
        ax.set_ylabel('MSE', fontsize=15)  # 设置y轴名称 y label
        ax.legend()  # 自动检测要在图例中显示的元素，并显示
        plt.savefig('../final_result/invivo_{}_test_MSE_curve.tif'.format(measure), dpi=600, facecolor='w',
                    edgecolor='w', transparent=False)
        plt.show()


if __name__ == "__main__":
    # model_list = ['tps16', 'tps25', 'sgan3', 'sgan2']
    # model_list = ['sgan2', 'diffusion_sgan2', 'sgan3', 'aanet', 'vae']
    plot_robust_figs()
