import numpy as np
import time
import torch
import random
from scipy.stats import qmc
from scipy.integrate import odeint
from utils_pinn import *

model_name = "MAWNO_MPF"


class Parameters:
    k_1_ = 0.01
    k_2_ = 0.01
    k_2__ = 10  # k''_2
    k_25_ = 0.04
    k_25__ = 100
    k_Wee = 1.5
    k_INH = 0.1
    k_CAK = 1
    G = 1 + k_INH / k_CAK


class TrainArgs:
    iteration = 500000
    epoch_step = 1000
    test_step = epoch_step * 1
    initial_lr = 0.001
    main_path = "."

    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.10
    early_stop_number = 3


class Config:

    def __init__(self):

        self.model_name = model_name
        self.curve_names = self.curve_names = ["U", "V"]
        self.time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = 88

        self.T = 90
        self.cf = 13
        self.T_N = int(2 ** self.cf)

        self.prob_dim = 2
        self.y0 = np.asarray([0.02595122, 0.35314558])  # 这是y的初值

        self.seed_t1 = 0
        self.seed_t2 = 1

        self.t = np.linspace(0, self.T, self.T_N)
        self.t2 = np.linspace(0, self.T - 0.01, self.T_N)
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)

        self.x_wno = torch.tensor(np.expand_dims(np.repeat(self.t[:, np.newaxis], self.prob_dim, axis=1), axis=0),
                                  dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.expand_dims(np.repeat(self.t[:, np.newaxis], self.prob_dim, axis=1), axis=0),
                                  dtype=torch.float32).to(self.device)
        self.x2 = torch.tensor(np.expand_dims(np.repeat(self.t2[:, np.newaxis], self.prob_dim, axis=1), axis=0),
                                  dtype=torch.float32).to(self.device)

        self.truth = odeint(self.pend, self.y0, self.t)
        self.truth2 = odeint(self.pend, self.y0, self.t2)

        self.double_skip = True
        self.embed_dim = 16
        self.depth = 4

        self.modes = 64
        # self.width = 16
        self.level = 8
        self.fc_map_dim = 128

        self.mode_T = 0
        self.mode = "zero"
        self.wave = "sym9"
        self.lambda2 = 100
        self.pinn = 0
        self.fno = 0
        self.wno = 1

        self.log_path = "./saves/mawno/{}/{}_{}/logs/start.txt".format(model_name, model_name, self.time_string)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_dir = "./saves/mawno/{}/{}_{}/tensorboard/".format(model_name, model_name, self.time_string)
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.figure_save_path_folder = "{0}/saves/mawno/{1}/{2}_{3}/figure/".\
            format(self.args.main_path, self.model_name, self.model_name, self.time_string)
        self.train_save_path_folder = "{0}/saves/mawno/{1}/{2}_{3}/train/".\
            format(self.args.main_path, self.model_name, self.model_name, self.time_string)
        self.loss_save_path_folder = "{0}/saves/mawno/{1}/{2}_{3}/loss/". \
            format(self.args.main_path, self.model_name, self.model_name, self.time_string)
        os.makedirs(os.path.dirname(self.loss_save_path_folder), exist_ok=True)
        os.makedirs(os.path.dirname(self.figure_save_path_folder), exist_ok=True)
        os.makedirs(os.path.dirname(self.train_save_path_folder), exist_ok=True)
        save_dir = "./saves/mawno/{}/{}_{}/model_path/".format(model_name, model_name, self.time_string)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        self.save_path = os.path.join(save_dir, "MPF_MAWNO_model.pth")

    def pend(self, y, t):
        # k = Parameters
        dydt = np.asarray([
            self.params.k_1_ / self.params.G - (self.params.k_2_ + self.params.k_2__ * y[0] ** 2 + self.params.k_Wee) * y[0] + (self.params.k_25_ + self.params.k_25__ * y[0] ** 2) * (
                        y[1] / self.params.G - y[0]),
            self.params.k_1_ - (self.params.k_2_ + self.params.k_2__ * y[0] ** 2) * y[1]
        ])
        return dydt
