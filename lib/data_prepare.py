import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)


def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None, pad_with_last_sample=False
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :2]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    if pad_with_last_sample:
        train_num_padding = (batch_size - (len(x_train) % batch_size)) % batch_size
        x_train_padding = np.repeat(x_train[-1:], train_num_padding, axis=0)
        y_train_padding = np.repeat(y_train[-1:], train_num_padding, axis=0)
        x_train = np.concatenate([x_train, x_train_padding], axis=0)
        y_train = np.concatenate([y_train, y_train_padding], axis=0)

        val_num_padding = (batch_size - (len(x_val) % batch_size)) % batch_size
        x_val_padding = np.repeat(x_val[-1:], val_num_padding, axis=0)
        y_val_padding = np.repeat(y_val[-1:], val_num_padding, axis=0)
        x_val = np.concatenate([x_val, x_val_padding], axis=0)
        y_val = np.concatenate([y_val, y_val_padding], axis=0)

        test_num_padding = (batch_size - (len(x_test) % batch_size)) % batch_size
        x_test_padding = np.repeat(x_test[-1:], test_num_padding, axis=0)
        y_test_padding = np.repeat(y_test[-1:], test_num_padding, axis=0)
        x_test = np.concatenate([x_test, x_test_padding], axis=0)
        y_test = np.concatenate([y_test, y_test_padding], axis=0)

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )  #torch.utils.data.TensorDataset对数据进行打包
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
