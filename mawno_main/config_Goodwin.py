import numpy as np
import time
import torch
import random
from scipy.stats import qmc
from utils_pinn import *
from scipy.integrate import odeint

model_name = "MAWNO_Goodwin"


class Parameters:
    # line 1
    a1 = 360
    A1 = 36
    b1 = 10
    k11 = 1
    k12 = 0
    alp1 = 0.5
    bet1 = 0
    a2 = 360
    A2 = 43
    b2 = 10
    k21 = 0
    k22 = 1
    alp2 = 0.6
    bet2 = 0


class TrainArgs:
    iteration = 20000
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
        self.curve_names = self.curve_names = ["x1", "x2", "y1","y2"]
        self.time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = 0

        self.T = 50
        self.cf = 11
        self.T_N = int(2 ** self.cf)
        # self.T_N = 10000

        self.prob_dim = 4
        self.y0 = np.asarray([0.5, 0.5, 0.5, 0.5])
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
        self.wave = "db6"
        self.lambda2 = 10
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

        self.save_path = os.path.join(save_dir, "GoodWin_MAWNO_model.pth")

    def pend(self, y, t):
        X1, X2, Y1, Y2 = y[0], y[1], y[2], y[3]

        dy0 = self.params.a1 / (
                    self.params.A1 + self.params.k11 * Y1 + self.params.k12 * Y2) - self.params.b1
        dy1 = self.params.a2 / (
                    self.params.A2 + self.params.k21 * Y1 + self.params.k22 * Y2) - self.params.b2
        dy2 = self.params.alp1 * X1 - self.params.bet1
        dy3 = self.params.alp2 * X2 - self.params.bet2

        dydt = np.asarray([dy0, dy1, dy2, dy3])

        return dydt

