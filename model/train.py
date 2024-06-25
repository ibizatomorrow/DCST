import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys



# sys.path.insert(0, sys.path[0]+"/../")
# sys.path.append("..")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.DCST import DCST

#Knowledge Distillation for METRLA
from STGNN.metrla.GWNet import T_GWNet
from STGNN.metrla.MTGNN import T_MTGNN
from STGNN.metrla.AGCRN import T_AGCRN
from STGNN.metrla.STGCN import T_STGCN
import STGNN.metrla.DCRNN.T_DCRNN as module_arch
from STGNN.metrla.DCRNN.util import load_graph_data
from parse_config import ConfigParser

# #Knowledge Distillation for PEMSBAY
# from STGNN.pemsbay.GWNet import T_GWNet
# from STGNN.pemsbay.MTGNN import T_MTGNN
# from STGNN.pemsbay.STGCN import T_STGCN
# import STGNN.pemsbay.DCRNN.T_DCRNN as module_arch
# from STGNN.pemsbay.DCRNN.util import load_graph_data
# from parse_config import ConfigParser
# from STGNN.pemsbay.AGCRN import T_AGCRN

# #Knowledge Distillation for PEMSD7
# from STGNN.pemsd7.GWNet import T_GWNet
# from STGNN.pemsd7.MTGNN import T_MTGNN
# from STGNN.pemsd7.AGCRN import T_AGCRN
# import STGNN.pemsd7.DCRNN.T_DCRNN as module_arch
# from STGNN.pemsd7.DCRNN.util import load_graph_data
# from parse_config import ConfigParser
# from STGNN.pemsd7.STGCN import T_STGCN


#nni
import nni
from nni.utils import merge_parameter

batch_num = 0 
cl_len = 1


# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch[...,0:1].to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out

@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)
    nni.report_final_result({'default': mae, 'mae': mae, 'rmse': rmse, "mape": mape})



def train_kd_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, kd_weight=0.5, if_cl = False, cl_steps=2000, log=None
):
    global cfg, global_iter_count, global_target_length
    global batch_num, cl_len

    #GWNet as teacher model
    teacher_model = T_GWNet.gwnet()
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.load_state_dict(torch.load("../STGNN/metrla/GWNet/metrla.pth"))
    teacher_model.eval()

    # #MTGNN as teacher model
    # teacher_model = T_MTGNN.mtgnn()
    # teacher_model = teacher_model.to(DEVICE)
    # teacher_model.load_state_dict(torch.load("../STGNN/metrla/MTGNN/metrla.pth"))
    # teacher_model.eval()

    # #AGCRN as teacher model
    # teacher_model = T_AGCRN.AGCRN()
    # teacher_model = teacher_model.to(DEVICE)
    # teacher_model.load_state_dict(torch.load("../STGNN/metrla/AGCRN/metrla.pth"))
    # teacher_model.eval()

    # #STGCN as teacher model
    # teacher_model = T_STGCN.STGCN()
    # teacher_model = teacher_model.to(DEVICE)
    # teacher_model.load_state_dict(torch.load("../STGNN/metrla/STGCN/metrla.tar")['model_state_dict'])
    # teacher_model.eval()

    # #DCRNN as teacher model
    # config = ConfigParser()
    # graph_pkl_filename = '../STGNN/metrla/DCRNN/data/adj_mx.pkl'
    # _, _, adj_mat = load_graph_data(graph_pkl_filename)
    # adj_arg = {"adj_mat": adj_mat}
    # teacher_model = config.initialize('arch', module_arch, **adj_arg)
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint['state_dict']
    # teacher_model.load_state_dict(state_dict)
    # teacher_model = teacher_model.to(DEVICE)
    # teacher_model.eval()


    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        batch_num = batch_num + 1

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        #student model
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        #GWNet teacher model
        with torch.no_grad():
            x_batch = x_batch[:,:,:,0:2]
            x_batch = x_batch.transpose(1,3)
            x_batch = nn.functional.pad(x_batch,(1,0,0,0)).to(DEVICE)

            out_teacher_batch = teacher_model(x_batch)
            out_teacher_batch = SCALER.inverse_transform(out_teacher_batch)

        # #MTGNN teacher model
        # with torch.no_grad():
        #     x_batch = x_batch[:,:,:,0:2]
        #     x_batch = x_batch.transpose(1,3)
        #     id = torch.tensor(np.arange(207)).to(DEVICE)

        #     out_teacher_batch = teacher_model(x_batch, idx=id)
        #     out_teacher_batch = SCALER.inverse_transform(out_teacher_batch)

        # #AGCRN teacher model
        # with torch.no_grad():
        #     x_batch = x_batch[:,:,:,0:1]

        #     out_teacher_batch = teacher_model(x_batch, y_batch)
        #     out_teacher_batch = SCALER.inverse_transform(out_teacher_batch)

        # # STGCN teacher model
        # with torch.no_grad():
        #     x_batch = x_batch[:,:,:,0:2]
        #     out_teacher_batch = teacher_model(x_batch)
        #     out_teacher_batch = SCALER.inverse_transform(out_teacher_batch)
        #     out_teacher_batch = out_teacher_batch.transpose(0, 1)
        #     out_teacher_batch = out_teacher_batch.unsqueeze(dim=-1)
        
        # # DCRNN teacher model
        # with torch.no_grad():
        #     x_batch = x_batch[:,:,:,0:2]
        #     y_batch[...,0:1] = SCALER.transform(y_batch[...,0:1])
        #     out_teacher_batch = teacher_model(x_batch, y_batch, 0)
        #     out_teacher_batch = SCALER.inverse_transform(out_teacher_batch)
        #     y_batch[...,0:1] = SCALER.inverse_transform(y_batch[...,0:1])
        #     out_teacher_batch = out_teacher_batch.transpose(0, 1)
        #     out_teacher_batch = out_teacher_batch.unsqueeze(dim=-1).cuda()

        #curriculum learning
        if if_cl:
            if batch_num % cl_steps == 0 and cl_len <= 12:
                cl_len += 1
            hard_loss = criterion(out_batch[:, :cl_len, :, :], y_batch[:, :cl_len, :,0:1])
            soft_loss = criterion(out_batch[:, :cl_len, :, :], out_teacher_batch[:, :cl_len, :, :])
        
        else:
            hard_loss = criterion(out_batch, y_batch[...,0:1])
            soft_loss = criterion(out_batch, out_teacher_batch)

        KD_loss = hard_loss*kd_weight + soft_loss*(1 - kd_weight)

        batch_loss_list.append(KD_loss.item())

        optimizer.zero_grad()
        KD_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss

def train_kd(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
    kd_weight=0.5,
    if_cl = False,
    cl_steps = 2000
):


    model = model.to(DEVICE)
    
    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_kd_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, kd_weight, if_cl, cl_steps, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                break
        nni.report_intermediate_result({'default': val_loss, 'Train_Loss': train_loss, 'Val_Loss': val_loss})
 

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)

    return model





if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="METRLA")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument('--kd_weight', type=int, default=0.5, help='KD weight')
    parser.add_argument('--if_cl', type=bool, default=False, help='if use curriculum learning')
    parser.add_argument('--cl_steps', type=int, default=2000, help='steps of curriculum learning')

    args = parser.parse_args()

    #nni
    nni_params = nni.get_next_parameter()
    args = merge_parameter(args, nni_params)

    #set random seed here
    seed = 3407 
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if_cl = args.if_cl
    cl_steps = args.cl_steps
    kd_weight = args.kd_weight
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = DCST.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = DCST(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY", "PEMSD7"):
        criterion = MaskedMAELoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- Distillation Model and Test--------------------------- #
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train_kd(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
        kd_weight = kd_weight,
        if_cl = if_cl,
        cl_steps = cl_steps,
    )
        
    print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, log=log)

    log.close()