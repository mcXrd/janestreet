import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire
from torch import Tensor
from resnet1d import Net1D

CSV_PATH = (
    "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/encoded.csv"
)
ENCODED_CSV = "encoded.csv"
ENC_ENABLED = True
print("ENC_ENABLED {}".format(ENC_ENABLED))


def strip_for_batch_size(df, batch_size):
    strip = 1 + df.shape[0] % batch_size
    return df.loc[: df.shape[0] - strip]


class JaneStreetDataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None, batch_size=None):
        df.insert(4, "trade", None)
        df.loc[df["resp"] <= 0, "trade"] = -10
        df.loc[df["resp"] > 0, "trade"] = 10
        df.trade = df.trade.multiply(1)
        df.resp = df.resp.multiply(1)
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.batch_size = batch_size

    # Getter
    def __getitem__(self, index):
        if ENC_ENABLED:
            x = (
                torch.tensor(self.df.iloc[index]["enc_feature_0":"enc_feature_49"])
                .float()
                .to(self.device)
            )
        else:
            x = (
                torch.tensor(self.df.iloc[index]["feature_0":"feature_129"])
                .float()
                .to(self.device)
            )

        sample = (
            x,
            torch.tensor(self.df.iloc[index]["trade":"trade"]).float().to(self.device),
        )
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


def get_core_model(input_size, output_size, hidden_count, dropout_p=0.15, net_width=32):
    if True:
        base_filters = 50
        filter_list = [64, 160, 160, 400, 400, 1024, 1024]
        m_blocks_list = [2, 2, 2, 3, 3, 4, 4]
        model = Net1D(
            in_channels=1,
            base_filters=base_filters,
            ratio=1.0,
            filter_list=filter_list,
            m_blocks_list=m_blocks_list,
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            n_classes=1,
        )
        return model

    assert hidden_count > 0
    layers = []

    def append_layer(layers, _input_size, _output_size, just_linear=False):
        layers.append(torch.nn.Linear(_input_size, _output_size))
        if just_linear:
            return
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(torch.nn.ReLU())
        layers.append(nn.BatchNorm1d(_output_size))

    append_layer(layers, input_size, net_width)

    for one in range(hidden_count):
        append_layer(layers, net_width, net_width)

    append_layer(layers, net_width, output_size, just_linear=True)
    return torch.nn.Sequential(*layers)


class CustomSmoothL1Loss(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = torch.clone(target)
        # where we have missmatch in signs - meaning we missed big - we will make the error way bigger
        target[input * target < 0] = target[input * target < 0] * 50
        ret = super().forward(input, target)
        return ret


def train_model(
    model,
    train_loader,
    validation_loader,
    n_epoch=None,
    lr=None,
    device=None,
    phase=None,
    criterion=None,
):
    writer = SummaryWriter()
    # criterion = nn.SmoothL1Loss()
    # criterion = CustomSmoothL1Loss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr * 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr)
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y.float())
            writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
            loss.backward()
            optimizer.step()
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())
        # perform a prediction on the validation data
        accurate_guess = 0
        total_count = 0
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())

            for one in range(len(z)):
                total_count += 1
                if (z[one][0] > 0 and y_test[one][0] == 10) or (
                    z[one][0] <= 0 and y_test[one][0] == -10
                ):
                    accurate_guess += 1

            loss = criterion(z, y_test.float())
            writer.add_scalar("Validation data phase: {}".format(phase), loss, epoch)
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())
        print("epoch_{} accuracy: {}".format(epoch, accurate_guess / total_count))

    model.eval()

    def build_model_output(loader):
        model_output = []
        for x, y in loader:
            z = model(x.float())
            for row in z.cpu().detach().numpy():
                model_output.append(row)
        return np.array(model_output)

    return (
        accuracy_list,
        loss_list,
        build_model_output(train_loader),
        build_model_output(validation_loader),
    )


def visual_control(model, batch_size, device, df):
    model.eval()
    index = np.random.randint(1, len(df))
    a = df.iloc[index]["enc_feature_0":"enc_feature_49"].values
    a = list(a)
    r = []
    for one in range(batch_size):
        r.append(a)
    r = np.array(r)
    model_input = torch.from_numpy(r).float().to(device)
    z = model(model_input)
    print("Model output")
    print(z[0])
    print("actual_data")
    print(df.iloc[index]["trade":"resp"])


def print_losses(accuracy_list, n_epoch, phase):
    print("phase {}".format(phase))
    print(
        "validation loss mean: {} {} {}".format(
            np.mean(accuracy_list[1]),
            np.mean(accuracy_list[int(n_epoch / 2)]),
            np.mean(accuracy_list[n_epoch - 1]),
        )
    )
    print(
        "validation loss variance: {} {} {}".format(
            np.var(accuracy_list[1]),
            np.var(accuracy_list[int(n_epoch / 2)]),
            np.var(accuracy_list[n_epoch - 1]),
        )
    )


def create_and_train_model(
    nrows=20000,  # 2390491 total
    big_number=1500,
    small_number=1500,
    dropout_p=0.15,
    validation_size=1000,
    batch_size=200,
    n_epoch=2,
    lr=0.05,
    effective_train_data=60000,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(
        CSV_PATH,
        nrows=nrows,
        skiprows=range(1, 500000),
    )
    before = df.shape[0]
    df = df[df["weight"] > 0]
    after = df.shape[0]
    print("after / before {}".format(after / before))
    df = df.fillna(0)
    frac = (1 / 20) * (before / after)
    n_epoch = int(effective_train_data / (df.shape[0] * frac)) + 1
    if n_epoch < 2:
        n_epoch = 2
    print("n_epoch {}".format(n_epoch))

    # df = df.multiply(1)
    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index()
    train_sample = strip_for_batch_size(train_sample, batch_size)
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index()
    validation_sample = strip_for_batch_size(validation_sample, batch_size)

    janestreet_train = JaneStreetDataset(df=train_sample, device=device)
    janestreet_validation = JaneStreetDataset(df=validation_sample, device=device)
    train_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet_train, batch_size=batch_size, drop_last=True
    )
    validation_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet_validation, batch_size=batch_size, drop_last=True
    )
    hidden_count = 2
    big_number = 1500
    if ENC_ENABLED:
        model = get_core_model(
            50, 1, hidden_count, dropout_p=0.15, net_width=big_number
        )
    else:
        model = get_core_model(
            130, 1, hidden_count, dropout_p=0.15, net_width=big_number
        )
    model = model.float()
    model.to(device)

    train_res = train_model(
        model,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
        phase=1,
    )
    (
        accuracy_list,
        loss_list,
        train_model_output,
        validation_model_output,
    ) = train_res
    assert len(train_model_output) == len(janestreet_train)
    assert len(validation_model_output) == len(janestreet_validation)

    print_losses(accuracy_list, n_epoch, 1)

    visual_control(
        model,
        batch_size,
        device,
        validation_sample,
    )
    return model, device


def main(
    nrows=20000,  # 2390491 total
    big_number=1500,
    small_number=1500,
    dropout_p=0.15,
    validation_size=500000,
    batch_size=200,
    n_epoch=2,
    lr=0.05,
    effective_train_data=100000,
):
    model, device = create_and_train_model(
        nrows=nrows,  # 2390491 total
        big_number=big_number,
        small_number=small_number,
        dropout_p=dropout_p,
        validation_size=validation_size,
        batch_size=batch_size,
        n_epoch=n_epoch,
        lr=lr,
        effective_train_data=effective_train_data,
    )
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
