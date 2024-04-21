import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime


def add_time(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "myprint":
            with open(args[1], "a") as f:
                f.write("{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))), end="")
        ret = func(*args, **kwargs)
        return ret

    return wrapper


@add_time
def myprint(string, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(string))
    print(string)


def write_start_log(config, time_string) -> None:
    log_path = config.log_path
    myprint("using {}".format(str(config.device)), log_path)  # 设备
    myprint("iteration = {}".format(config.args.iteration), log_path)  # 共运行轮数
    myprint("epoch_step = {}".format(config.args.epoch_step), log_path)  # 多少轮打印loss
    myprint("test_step = {}".format(config.args.test_step), log_path)  # 多少轮draw一次
    myprint("model_name = {}".format(config.model_name), log_path)  # 模型名称
    myprint("time_string = {}".format(time_string), log_path)  # 时间
    myprint("seed = {}".format(config.seed), log_path)  # 种子
    myprint("seed_t1 = {}".format(config.seed_t1), log_path)
    myprint("seed_t2 = {}".format(config.seed_t2), log_path)
    myprint("initial_lr = {}".format(config.args.initial_lr), log_path)  # 初始lr
    myprint("T_N = {}".format(config.T_N), log_path)  # T_N
    myprint("level = {}".format(config.level), log_path)  # level
    myprint("wave = {}".format(config.wave), log_path)  # wave
    myprint("mode_T = {}".format(config.mode_T), log_path)  # mode
    myprint("mode = {}".format(config.mode), log_path)  # mode
    myprint("depth = {}".format(config.depth), log_path)
    myprint("double_skip = {}".format(config.double_skip), log_path)
    myprint("embed_dim = {}".format(config.embed_dim), log_path)
    myprint("lambda2 = {}".format(config.lambda2), log_path)  # loss2权值

    # myprint("cyclic = {}".format(config.cyclic), config.args.log_path)  # 是否循环
    # myprint("stable = {}".format(config.stable), config.args.log_path)  # 是否stable
    # myprint("derivative = {}".format(config.derivative), config.args.log_path)
    # myprint("activation = {}".format(config.activation), config.args.log_path)
    # myprint("boundary = {}".format(config.boundary), config.args.log_path)


def train_info(config, time_string, epoch_tmp, loss_record, real_loss_mse_record,
               real_loss_nmse_record,test_loss_nmse_record, time_record, y):
    train_info = {  # 为了保存训练信息
        "model_name": config.model_name,
        "seed": config.seed,
        "prob_dim": config.prob_dim,
        # "activation": config.activation,
        # "cyclic": self.config.cyclic,
        # "stable": self.config.stable,
        # "derivative": self.config.derivative,
        # "loss_average_length": self.config.loss_average_length,
        "epoch": config.args.iteration,
        "epoch_stop": epoch_tmp,
        "initial_lr": config.args.initial_lr,
        "loss_length": len(loss_record),
        "loss": np.asarray(loss_record),
        "real_loss_mse": np.asarray(real_loss_mse_record),
        "real_loss_nmse": np.asarray(real_loss_nmse_record),
        "test_loss_nmse": np.asarray(test_loss_nmse_record),
        "time": np.asarray(time_record),
        "y_predict": y.cpu().detach().numpy(),  # y[0, :, :].cpu().detach().numpy()
        "y_truth": np.asarray(config.truth),
        "y_shape": config.truth.shape,
        # "config": self.config,
        "time_string": time_string
    }
    print("successful to write")
    return train_info


def write_finish_log(config, finish_path, time_string, time_record_tmp, min_test_loss_nmse,min_real_loss_nmse) -> None:
    with open(finish_path, "a") as f:
        f.write(
            "model_name:\t{0}\ntime_string:\t{1}\nseed:\t{2}\ntime_record_tmp[-1]:\t{3:.2f}\niteration:\t{4}\ninitial_lr:\t{"
            "5:.6f}\nmin_test_nmse:\t{6}\nmin_real_nmse:\t{7}\npinn:\t{8}\n".format(
                config.model_name,  # 0
                time_string,  # 1
                config.seed,  # 2
                time_record_tmp[-1] / 60.0,  # 3
                config.args.iteration,  # 4
                config.args.initial_lr,  # 5
                min_test_loss_nmse,  # 6
                min_real_loss_nmse,  # 7
                # sum(self.loss_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 6
                # sum(self.real_loss_mse_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,
                # # 7
                # sum(self.real_loss_nmse_record_tmp[
                #     -self.config.loss_average_length:]) / self.config.loss_average_length,  # 8
                config.pinn,  # 9
                # self.config.activation,  # 10
                # self.config.stable,  # 11
                # self.config.cyclic,  # 12
                # self.config.derivative,  # 13
                # self.config.boundary,  # 14
                # self.config.loss_average_length,  # 15
                # "{}-{}".format(self.config.args.iteration - self.config.loss_average_length,
                #                self.config.args.iteration),  # 16
                # self.config.init_weights,  # 17
                # self.config.init_weights_strategy,  # 18
                # self.config.scheduler,  # 19
                # self.activation_weights_record[0][-1][0] if self.config.activation in ["adaptive_5",
                #                                                                        "adaptive_6"] else None,  # 20
                # self.activation_weights_record[0][-1][1] if self.config.activation in ["adaptive_5",
                #                                                                        "adaptive_6"] else None,  # 21
                # self.activation_weights_record[0][-1][2] if self.config.activation in ["adaptive_5",
                #                                                                        "adaptive_6"] else None,  # 22
                # self.activation_weights_record[0][-1][3] if self.config.activation in ["adaptive_5",
                #                                                                        "adaptive_6"] else None,  # 23
                # self.activation_weights_record[0][-1][4] if self.config.activation in ["adaptive_5",
                #                                                                        "adaptive_6"] else None,  # 24
                # self.activation_weights_record[0][-1][5] if self.config.activation in ["adaptive_6"] else None,  # 25
            ))


def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        legend_fontsize=15,
        fig_title=None,
        legend_loc="best",
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=10,
        fig_grid=False,
        y_scale=False,
        marker_size=0,
        line_width=2,
        x_label_size=10,
        y_label_size=10,
        number_label_size=10,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        tight_layout_flag=True,
        x_ticks_dress=None,
        y_ticks_dress=None
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i],
                 linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        if x_ticks_dress:
            plt.xticks(x_ticks, x_ticks_dress)
        else:
            plt.xticks(x_ticks)
    if y_ticks_set_flag:
        if y_ticks_dress:
            plt.xticks(y_ticks, y_ticks_dress)
        else:
            plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                       ncol=legend_ncol, loc=legend_loc)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize, loc=legend_loc)
    if y_scale:
        plt.yscale('log')
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_biology_figure(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        legend_fontsize=15,
        fig_title=None,
        legend_loc="best",
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=10,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=10,
        y_label_size=10,
        number_label_size=10,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        tight_layout_flag=True,
        x_ticks_dress=None,
        y_ticks_dress=None,
        y_lim=None
) -> None:

    # if y_lim is None:
    #     y_lim = [0, 5]
    assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i],
                 linestyle=line_style_list[i])
    # plt.xlabel(fig_x_label, fontsize=x_label_size)
    # plt.ylabel(fig_y_label, fontsize=y_label_size)
    # plt.xticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    if y_ticks_set_flag:
        if y_ticks_dress:
            plt.xticks(y_ticks, y_ticks_dress)
        else:
            plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                       ncol=legend_ncol, loc=legend_loc)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize, loc=legend_loc)
    # plt.grid(axis='y')
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if y_lim:
        plt.ylim(y_lim)
    if fig_grid:
        plt.grid(axis='y')
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_different_x(
        y_lists,
        x_lists,
        color_list,
        line_style_list,
        legend_list=None,
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        tight_layout_flag=True
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_lists: (list[list]) x value of lines. e.g., [[1,2,3,4], [5,6,7]]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i],
                 linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        plt.xticks(x_ticks)
    if y_ticks_set_flag:
        plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                       ncol=legend_ncol)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def smooth_conv(data, kernel_size: int = 10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def draw_multiple_loss(
        loss_path_list,
        color_list,
        line_style_list,
        legend_list,
        fig_title,
        start_index,
        end_index,
        threshold=None,
        smooth_kernel_size=1,
        marker_size=0,
        line_width=1,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        show_flag=True,
        save_flag=False,
        save_path=None,
        only_original_flag=False,
        fig_x_label="epoch",
        fig_y_label="loss",
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        tight_layout_flag=True
):
    # line_n = len(loss_path_list)
    assert (len(loss_path_list) if only_original_flag else 2 * len(loss_path_list)) == len(color_list) == len(
        line_style_list) == len(
        legend_list), "Note that for each loss in loss_path_list, this function will generate an original version and a smoothed version. So please give the color_list, line_style_list, legend_list for all of them"
    x_list = range(start_index, end_index)
    y_lists = [np.load(one_path) for one_path in loss_path_list]
    print("length:", [len(item) for item in y_lists])
    y_lists_smooth = [smooth_conv(item, smooth_kernel_size) for item in y_lists]
    for i, item in enumerate(y_lists):
        print("{}: {}".format(legend_list[i], np.mean(item[start_index: end_index])))
        if threshold:
            match_index_list = np.where(y_lists_smooth[i] <= threshold)
            if len(match_index_list[0]) == 0:
                print("No index of epoch matches condition '< {}'!".format(threshold))
            else:
                print("Epoch {} is the first value matches condition '< {}'!".format(match_index_list[0][0], threshold))
    y_lists = [item[x_list] for item in y_lists]
    y_lists_smooth = [item[x_list] for item in y_lists_smooth]
    draw_two_dimension(
        y_lists=y_lists if only_original_flag else y_lists + y_lists_smooth,
        x_list=x_list,
        color_list=color_list,
        legend_list=legend_list,
        legend_location=legend_location,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_ncol=legend_ncol,
        line_style_list=line_style_list,
        fig_title=fig_title,
        fig_size=fig_size,
        fig_x_label=fig_x_label,
        fig_y_label=fig_y_label,
        x_ticks_set_flag=x_ticks_set_flag,
        y_ticks_set_flag=y_ticks_set_flag,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        marker_size=marker_size,
        line_width=line_width,
        show_flag=show_flag,
        save_flag=save_flag,
        save_path=save_path,
        tight_layout_flag=tight_layout_flag
    )


class MultiSubplotDraw:
    def __init__(self, row, col, fig_size=(8, 6), show_flag=True, save_flag=False, save_path=None, save_dpi=300,
                 tight_layout_flag=False):
        self.row = row
        self.col = col
        self.subplot_index = 0
        self.show_flag = show_flag
        self.save_flag = save_flag
        self.save_path = save_path
        self.save_dpi = save_dpi
        self.tight_layout_flag = tight_layout_flag
        self.fig = plt.figure(figsize=fig_size)

    def draw(self, ):
        if self.tight_layout_flag:
            plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_path, dpi=self.save_dpi)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def add_subplot(
            self,
            y_lists,
            x_list,
            color_list,
            line_style_list,
            legend_list=None,
            legend_location="auto",
            legend_bbox_to_anchor=(0.515, 1.11),
            legend_ncol=3,
            legend_fontsize=15,
            fig_title=None,
            fig_x_label="time",
            fig_y_label="val",
            fig_title_size=20,
            fig_grid=False,
            marker_size=0,
            line_width=2,
            x_label_size=15,
            y_label_size=15,
            number_label_size=15,
            x_ticks_set_flag=False,
            x_ticks=None,
            y_ticks_set_flag=False,
            y_ticks=None,
            scatter_period=0,
            scatter_marker=None,
            scatter_marker_size=None,
            scatter_marker_color=None
    ):
        # assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
        assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
        y_count = len(y_lists)
        self.subplot_index += 1
        ax = self.fig.add_subplot(self.row, self.col, self.subplot_index)
        for i in range(y_count):
            draw_length = min(len(x_list), len(y_lists[i]))
            # print("x_list[:draw_length]", x_list[:draw_length])
            # print("y_lists[i][:draw_length]", y_lists[i][:draw_length])
            # print("color_list[i]", color_list[i])
            ax.plot(x_list[:draw_length], y_lists[i][:draw_length], markersize=marker_size, linewidth=line_width,
                    c=color_list[i], linestyle=line_style_list[i], label=legend_list[i] if legend_list else None)
            if scatter_period > 0:
                scatter_x = [x_list[:draw_length][idx] for idx in range(len(x_list[:draw_length])) if
                             idx % scatter_period == 0]
                scatter_y = [y_lists[i][:draw_length][idx] for idx in range(len(y_lists[i][:draw_length])) if
                             idx % scatter_period == 0]
                print(scatter_x)
                print(scatter_y)
                ax.scatter(x=scatter_x, y=scatter_y, s=scatter_marker_size, c=scatter_marker_color,
                           marker=scatter_marker, linewidths=0, zorder=10)
        ax.set_xlabel(fig_x_label, fontsize=x_label_size)
        ax.set_ylabel(fig_y_label, fontsize=y_label_size)
        if x_ticks_set_flag:
            ax.set_xticks(x_ticks)
        if y_ticks_set_flag:
            ax.set_yticks(y_ticks)
        if legend_list:
            if legend_location == "fixed":
                ax.legend(fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                          ncol=legend_ncol)
            else:
                ax.legend(fontsize=legend_fontsize)
        if fig_title:
            ax.set_title(fig_title, fontsize=fig_title_size)
        if fig_grid:
            ax.grid(True)
        plt.tick_params(labelsize=number_label_size)
        return ax

    def add_subplot_turing(
            self,
            matrix,
            v_max,
            v_min,
            fig_title=None,
            fig_title_size=20,
            number_label_size=15,
    ):
        self.subplot_index += 1
        ax = self.fig.add_subplot(self.row, self.col, self.subplot_index)
        im1 = ax.imshow(matrix, cmap=plt.cm.jet, vmax=v_max, vmin=v_min, aspect='auto')
        ax.set_title(fig_title, fontsize=fig_title_size)
        plt.colorbar(im1, shrink=1)
        plt.tick_params(labelsize=number_label_size)
        return ax


"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(X, Y, Z)
plt.draw()
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(np.asarray([[x, y, z] for x, y, z in zip(X, Y, Z)]))
plt.show()
plt.clf()"""


def draw_three_dimension(
        lists,
        color_list,
        line_style_list,
        legend_list=None,
        fig_title=None,
        fig_x_label="X",
        fig_y_label="Y",
        fig_z_label="Z",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        lim_adaptive_flag=False,
        x_lim=(-25, 25),
        y_lim=(-25, 25),
        z_lim=(0, 50),
        line_width=1,
        alpha=1,
        x_label_size=15,
        y_label_size=15,
        z_label_size=15,
        number_label_size=15,
        fig_size=(8, 6),
        tight_layout_flag=True,
) -> None:
    for one_list in lists:
        assert len(one_list) == 3, "3D data please!"
        assert len(one_list[0]) == len(one_list[1]) == len(one_list[2]), "Dimension of X, Y, Z should be the same"
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    for i, one_list in enumerate(lists):
        ax.plot(one_list[0], one_list[1], one_list[2], linewidth=line_width, alpha=alpha, c=color_list[i],
                linestyle=line_style_list[i], label=legend_list[i] if legend_list else None)
    ax.legend(loc="lower left")
    ax.set_xlabel(fig_x_label, fontsize=x_label_size)
    ax.set_ylabel(fig_y_label, fontsize=y_label_size)
    ax.set_zlabel(fig_z_label, fontsize=z_label_size)
    if lim_adaptive_flag:
        x_lim = (min([min(item[0]) for item in lists]), max([max(item[0]) for item in lists]))
        y_lim = (min([min(item[1]) for item in lists]), max([max(item[1]) for item in lists]))
        z_lim = (min([min(item[2]) for item in lists]), max([max(item[2]) for item in lists]))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    if fig_title:
        ax.set_title(fig_title, fontsize=fig_title_size)
    plt.tick_params(labelsize=number_label_size)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class ColorCandidate:
    def __init__(self):
        self.raw_rgb = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 128, 0),
            (255, 127, 0),
            (255, 0, 127),
            (0, 128, 127),
            (150, 10, 100),
            (150, 50, 20),
            (100, 75, 20),
            (127, 128, 0),
            (127, 0, 255),
            (0, 64, 255),
            (20, 75, 100),
            (20, 50, 150),
            (100, 10, 150),
        ]

    @staticmethod
    def lighter(color_pair, rate=0.5):
        return [int(color_pair[i] + (255 - color_pair[i]) * rate) for i in range(3)]

    def get_color_list(self, n, light_rate=0.5):
        assert n <= 15
        return [self.encode(item) for item in self.raw_rgb[:n]] + [self.encode(self.lighter(item, light_rate)) for item
                                                                   in self.raw_rgb[:n]]

    @staticmethod
    def decode(color_str):
        return [int("0x" + color_str[2 * i + 1: 2 * i + 3], 16) for i in range(3)]

    @staticmethod
    def encode(color_pair):
        return "#" + "".join([str(hex(item))[2:].zfill(2) for item in color_pair])


if __name__ == "__main__":
    # T = 10.0
    # T_unit = 1e-5
    # X_start = 6.0
    # Y_start = 6.0
    # Z_start = 15.0
    #
    # rho = 14#28.0
    # sigma = 10#10.0
    # beta = 2.667#8.0 / 3.0
    #
    # X = [X_start]
    # Y = [Y_start]
    # Z = [Z_start]
    #
    #
    # for i in range(1, int(T / T_unit) + 1):
    #     X_old, Y_old, Z_old = X[-1], Y[-1], Z[-1]
    #
    #     X_new = X_old + (sigma * (Y_old - X_old) ) * T_unit
    #     Y_new = Y_old + (X_old * (rho - Z_old) - Y_old ) * T_unit
    #     Z_new = Z_old + (X_old * Y_old - beta * Z_old ) * T_unit
    #     X.append(X_new)
    #     Y.append(Y_new)
    #     Z.append(Z_new)
    # X = np.asarray(X)
    # Y = np.asarray(Y)
    # Z = np.asarray(Z)
    # print(list(X[:10]), "...", list(X[-10:]))
    # print(list(Y[:10]), "...", list(Y[-10:]))
    # print(list(Z[:10]), "...", list(Z[-10:]))

    # offset = 2
    # draw_three_dimension(
    #     lists=[[X, Y, Z], [X+offset, Y+offset, Z+offset]],
    #     legend_list=["true", "pred"],
    #     color_list=["r", "b"],
    #     line_style_list=["dashed", "solid"],
    #     fig_title="test",
    #     alpha=0.7,
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None,
    #     fig_size=(8, 6),
    #     line_width=0.5,
    #     lim_adaptive_flag=True
    # )
    #
    # print(min(X), max(X))
    # print(min(Y), max(Y))
    # print(min(Z), max(Z))

    # x_list = range(10000)
    # y_lists = [
    #     [0.005 * i + 10 for i in x_list],
    #     [-0.005 * i - 30 for i in x_list],
    #     [0.008 * i - 10 for i in x_list],
    #     [-0.006 * i - 20 for i in x_list],
    #     [-0.001 * i - 5 for i in x_list],
    #     [-0.003 * i - 1 for i in x_list]
    # ]
    # color_list = ["red", "blue", "green", "cyan", "black", "purple"]
    # line_style_list = ["dashed", "dotted", "dashdot", "dashdot", "dashdot", "dashdot"]
    # legend_list = ["red line", "blue line", "green line", "cyan line", "black line", "purple line"]
    # #
    # draw_two_dimension(
    #     y_lists=y_lists,
    #     x_list=x_list,
    #     color_list=color_list,
    #     legend_list=legend_list,
    #     line_style_list=line_style_list,
    #     fig_title=None,
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None,
    #     legend_location="fixed",
    #     legend_ncol=3,
    #     legend_bbox_to_anchor=(0.86, 1.19),
    #     tight_layout_flag=True
    # )

    x_list = range(10000)
    y_lists = [
        [0.005 * i + 10 for i in x_list],
        [-0.005 * i - 30 for i in x_list],
        # [0.008 * i - 10 for i in x_list],
        # [-0.006 * i - 20 for i in x_list],
        # [-0.001 * i - 5 for i in x_list],
        # [-0.003 * i - 1 for i in x_list]
    ]

    m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 7), tight_layout_flag=True)

    # truth = np.load("turing_truth.npy")
    # u = truth[-1, :, :, 0]
    # print(u.shape)
    # v = truth[-1, :, :, 1]
    # m.add_subplot_turing(
    #     matrix=u,
    #     v_max=u.max(),
    #     v_min=u.min(),
    #     fig_title="u")
    # m.add_subplot_turing(
    #     matrix=v,
    #     v_max=v.max(),
    #     v_min=v.min(),
    #     fig_title="v")
    # m.draw()

    ax = m.add_subplot(
        y_lists=y_lists,
        x_list=x_list,
        color_list=["yellow", "b"],
        legend_list=["111", "222"],
        line_style_list=["dashed", "solid"],
        fig_title="hello world",
        scatter_period=1000,
        scatter_marker="X",
        scatter_marker_size=100,
        scatter_marker_color="red"
    )
    ax.set_title("fuck", fontsize=15)
    m.add_subplot(
        y_lists=y_lists,
        x_list=x_list,
        color_list=["g", "grey"],
        legend_list=["111", "222"],
        line_style_list=["dashed", "solid"],
        fig_title="hello world")

    m.draw()

    # color_list = ["red", "blue"]#, "green", "cyan", "black", "purple"]
    # line_style_list = ["dashed", "dotted"]#, "dashdot", "dashdot", "dashdot", "dashdot"]
    # legend_list = ["red line", "blue line"]#, "green line", "cyan line", "black line", "purple line"]
    # #
    # draw_two_dimension(
    #     y_lists=y_lists,
    #     x_list=x_list,
    #     color_list=color_list,
    #     legend_list=legend_list,
    #     line_style_list=line_style_list,
    #     fig_title=None,
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None,
    #     legend_location="fixed",
    #     legend_ncol=3,
    #     legend_bbox_to_anchor=(0.515, 1.11),#(0.86, 1.19),
    #     tight_layout_flag=True
    # )

    # fig = plt.figure(figsize=(16, 6))
    # ax = fig.add_subplot(1, 3, 1)
    # ax.plot(x_list, [0.005 * i + 10 for i in x_list])
    # ax.legend(["aaa"])
    # ax.set_xlabel("hello", fontsize=20)
    # ax.set_xticks(range(0, 10001, 2000))
    # ax.grid(True)
    #
    # ax = fig.add_subplot(1, 3, 2)
    # ax.plot(x_list, [-0.005 * i - 30 for i in x_list])
    #
    # ax = fig.add_subplot(1, 3, 3)
    # ax.plot(x_list, [-0.005 * i - 30 for i in x_list])
    #
    # plt.tick_params(labelsize=20)
    # plt.show()
    # plt.close()

    # import numpy as np
    # x1 = [i * 0.01 for i in range(1000)]
    # y1 = [np.sin(i * 0.01) for i in range(1000)]
    # x2 = [i * 0.01 for i in range(500)]
    # y2 = [np.cos(i * 0.01) for i in range(500)]
    # x3 = []
    # y3 = []
    # draw_two_dimension_different_x(
    #     y_lists=[y1, y2, y3],
    #     x_lists=[x1, x2, x3],
    #     color_list=["b", "r", "y"],
    #     legend_list=["sin", "cos", "balabala"],
    #     line_style_list=["dotted", "dotted", "dotted"],
    #     fig_title="Anonymous",
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None
    # )

    # data = np.load("loss/SimpleNetworkSIRAges_Truth_100000_1000_0.01_2022-06-05-20-54-55_loss_100000.npy")
    # print(type(data))
    # print(data)
    # data_10 = smooth_conv(data, 500)
    # print(data_10)
    # data_20 = smooth_conv(data, 1000)
    # print(data_20)
    # start_index = 40000
    # end_index = 100000
    #
    # draw_two_dimension(
    #     y_lists=[data[start_index: end_index], data_10[start_index: end_index], data_20[start_index: end_index]],
    #     x_list=range(start_index, end_index),
    #     color_list=["r", "g", "b"],
    #     legend_list=["None", "10", "20"],
    #     line_style_list=["solid"] * 3,
    #     fig_title="Anonymous",
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None
    # )

    # a = np.asarray([1,2,4,5,7, 0])
    # print(np.where(a > 3))
