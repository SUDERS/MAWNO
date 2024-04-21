import numpy as np
import torch
import torch.fft
import pickle
from torch.utils.tensorboard import SummaryWriter
from utils_pinn import *
from mawnonet import MAWNONet
from config_Brusselator import Config as config


class train:
    def __init__(self, config, model):
        self.config = config
        self.criterion = torch.nn.MSELoss(reduction="mean").to(self.config.device)
        self.criterion_non_reduce = torch.nn.MSELoss(reduction="none").to(self.config.device)
        self.model = model
        self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey",
                               "teal", "olive"]

    def ode_gradient(self, x, y):
        k = self.config.params

        X = y[0, :, 0]
        Y = y[0, :, 1]

        X_t = torch.gradient(X, spacing=(self.config.t_torch,))[0]
        Y_t = torch.gradient(Y, spacing=(self.config.t_torch,))[0]

        F_X = X_t - (k.A+X*X*Y-k.B*X-X)
        F_Y = Y_t - (k.B*X-X*X*Y)
        return F_X, F_Y

    def loss(self, y):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_1, ode_2 = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.tensor([0.0] * self.config.T_N).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = self.config.lambda2 * (self.criterion(ode_1, zeros_1D))
        loss3 = self.config.lambda2 * (self.criterion(ode_2, zeros_1D))

        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list

    def real_loss(self, y):

        truth = torch.tensor(self.config.truth[:, :]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, :], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, :], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

    def test_loss(self,y2):
        truth2 = torch.tensor(self.config.truth2[:, :]).to(self.config.device)
        test_loss_nmse = torch.mean(self.criterion_non_reduce(y2[0, :, :], truth2) / (truth2 ** 2))
        return test_loss_nmse

    def early_stop(self):
        if not self.config.args.early_stop or len(
                self.test_loss_nmse_record_tmp) < 2 * self.config.args.early_stop_period:
            return False
        sum_old = sum(
            self.test_loss_nmse_record_tmp[
            - 2 * self.config.args.early_stop_period: - self.config.args.early_stop_period])
        sum_new = sum(self.test_loss_nmse_record_tmp[- self.config.args.early_stop_period:])

        train_sum_old = sum(
            self.loss_record_tmp[- 2 * self.config.args.early_stop_period: - self.config.args.early_stop_period])
        train_sum_new = sum(self.loss_record_tmp[- self.config.args.early_stop_period:])
        if (sum_new - sum_old) / sum_old >= - self.config.args.early_stop_tolerance and (
                (sum_new - sum_old) / sum_old) <= 0:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.test_loss_nmse_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.test_loss_nmse_record_tmp) - self.config.args.early_stop_period,
                len(self.test_loss_nmse_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.log_path)
            print("train_reduces {0:.4f}", (train_sum_old - train_sum_new) / train_sum_old)
            self.config.args.early_stop_number -= 1
            print("early_stop_number: ", self.config.log_path)
        else:

            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.test_loss_nmse_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.test_loss_nmse_record_tmp) - self.config.args.early_stop_period,
                len(self.test_loss_nmse_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.log_path)
            print("tain_reduces {0:.4f}", (train_sum_old - train_sum_new) / train_sum_old)
            return False
        if self.config.args.early_stop_number == 0:
            myprint("[Early Stop] Early Stop!", self.config.log_path)
            return True

    def train_model(self):
        writer = SummaryWriter(log_dir=self.config.log_dir)
        step_size = 10000
        gamma = 0.5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 20000 + 1))
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 50000], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)

        self.y_tmp = None
        self.epoch_tmp = None
        self.loss_record_tmp = None
        self.real_loss_mse_record_tmp = None
        self.real_loss_nmse_record_tmp = None
        self.test_loss_nmse_record_tmp = None
        self.time_record_tmp = None

        start_time = time.time()
        start_time_0 = start_time
        loss_record = []
        real_loss_mse_record = []
        real_loss_nmse_record = []
        test_loss_nmse_record = []
        time_record = []

        write_start_log(self.config, self.config.time_string)

        for epoch in range(1, self.config.args.iteration + 1):
            self.model.train()
            optimizer.zero_grad()
            y = self.model(self.config.x)
            loss, loss_list = self.loss(y)
            real_loss_mse, real_loss_nmse = self.real_loss(y)
            if epoch % 100 == 0:
                loss_record.append(loss.item())
            real_loss_mse_record.append(real_loss_mse.item())
            real_loss_nmse_record.append(real_loss_nmse.item())

            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train_loss', loss.item(), epoch)
            writer.add_scalar('train_real_nmse_loss', real_loss_nmse.item(), epoch)
            writer.add_scalar('learning_rate', current_lr, epoch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.model.eval()
            y2 = self.model(self.config.x2)
            test_loss_nmse = self.test_loss(y2)
            test_loss_nmse_record.append(test_loss_nmse.item())

            now_time = time.time()
            time_record.append(now_time - start_time_0)

            writer.add_scalar('test_NMSE_loss', test_loss_nmse.item(), epoch)

            if epoch % self.config.args.epoch_step == 0:
                now_time = time.time()

                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                print(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss.item(), loss_print_part,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                                                         (now_time - start_time_0) / 60.0 / epoch * (
                                                                     self.config.args.iteration - epoch)))
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.12f} {3} TEST_NMSE-Loss: {4:.12f} real_NMSE-Loss: {5:.12f} Lr:{6:.12f} Time:{7:.6f}s ({8:.2f}min in total, {9:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss.item(), loss_print_part, test_loss_nmse.item(),
                        real_loss_nmse.item(),
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                                                         (now_time - start_time_0) / 60.0 / epoch * (
                                                                 self.config.args.iteration - epoch)),
                    self.config.log_path)

                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp = y
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.real_loss_mse_record_tmp = real_loss_mse_record
                    self.real_loss_nmse_record_tmp = real_loss_nmse_record
                    self.test_loss_nmse_record_tmp = test_loss_nmse_record
                    self.time_record_tmp = time_record
                    self.test_model()

                    myprint("saving training info ...", self.config.log_path)
                    train_info1 = train_info(self.config, self.config.time_string, self.epoch_tmp, self.loss_record_tmp,
                                             self.real_loss_mse_record_tmp, self.real_loss_nmse_record_tmp,
                                             self.test_loss_nmse_record_tmp,
                                             time_record, y[0, :, :])

                    train_info_path_loss = "{}/{}_{}_info.npy".format(self.config.train_save_path_folder,
                                                                      self.config.model_name, self.config.time_string)
                    with open(train_info_path_loss, "wb") as f:
                        pickle.dump(train_info1, f)
                    if epoch >= 3 * self.config.args.test_step:
                        self.early = self.early_stop()
                    myprint("real_nmse_loss_last:{}".format(np.sum(real_loss_nmse_record[-5:]) / 5),
                            self.config.log_path)
                    myprint("test_nmse_loss_last:{}".format(np.sum(test_loss_nmse_record[-5:]) / 5),
                            self.config.log_path)
                #                     print("early:",self.early)

                if (epoch % (self.config.args.iteration / 4)) == 0 or self.early:
                    finish_path = "{0}/saves/mawno/{1}/{2}_{3}/logs/finish_logs.txt".format(
                        self.config.args.main_path, self.config.model_name,
                        self.config.model_name, self.config.time_string)
                    os.makedirs(os.path.dirname(finish_path), exist_ok=True)
                    min_loss_nmse = np.min(test_loss_nmse_record)
                    min_real_loss_nmse = np.min(real_loss_nmse_record)
                    write_finish_log(self.config, finish_path, self.config.time_string, self.time_record_tmp,
                                     min_loss_nmse, min_real_loss_nmse)

                    myprint("figure_save_path_folder: {}".format(self.config.figure_save_path_folder),
                            self.config.log_path)
                    myprint("train_save_path_folder: {}".format(self.config.train_save_path_folder),
                            self.config.log_path)
                    myprint("Finished.", self.config.log_path)
                    if epoch == self.config.args.iteration or self.early:
                        writer.close()
                    if self.early:
                        break

    def test_model(self):
        y_draw = self.y_tmp[0].cpu().detach().numpy().swapaxes(0, 1)
        x_draw = self.config.t
        y_draw_truth = self.config.truth.swapaxes(0, 1)
        save_path = "{}/{}_{}_epoch={}.png".format(self.config.figure_save_path_folder, self.config.model_name,
                                                   self.config.time_string, self.epoch_tmp)
        draw_two_dimension(
            y_lists=np.concatenate([y_draw, y_draw_truth], axis=0),
            x_list=x_draw,
            color_list=self.default_colors[: 2 * self.config.prob_dim],
            legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
            line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
            fig_title="{}_{}_epoch={}".format(self.config.model_name, self.config.time_string, self.epoch_tmp),
            fig_size=(8, 6),
            legend_fontsize=10,
            show_flag=False,
            save_flag=True,
            save_path=save_path,
            save_dpi=300,
            legend_loc="center right",
        )
        print("Figure is saved to {}".format(save_path))
        self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25], "Train_MSE")


    def draw_loss_multi(self, loss_list, last_rate_list, character):
        save_path = "{}/{}_{}_epoch={}.png".format(self.config.loss_save_path_folder, self.config.model_name,
                                                   self.config.time_string, self.epoch_tmp)
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=False, save_flag=True, save_path=save_path)
        for one_rate in last_rate_list:
            m.add_subplot(
                y_lists=[loss_list[-int(len(loss_list) * one_rate):]],
                x_list=range(len(loss_list) - int(len(loss_list) * one_rate) + 1, len(loss_list) + 1),
                color_list=["blue"],
                line_style_list=["solid"],
                fig_title="{}_Loss - lastest ${}$% - epoch ${}$ to ${}$".format(character, int(100 * one_rate),
                                                                                len(loss_list) - int(
                                                                                    len(loss_list) * one_rate) + 1,
                                                                                len(loss_list)),
                fig_x_label="epoch",
                fig_y_label="loss")
        m.draw()



if __name__ == '__main__':
    config = config()
    print(config.truth[400])
    model = MAWNONet(config, embed_dim=config.embed_dim, depth=config.depth)
    model.to(config.device)
    print(model.state_dict().keys())
    train = train(config, model)
    train.train_model()

    torch.save(model.state_dict(), config.save_path)

