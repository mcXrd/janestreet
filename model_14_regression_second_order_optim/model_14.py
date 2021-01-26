import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire
from torch import Tensor

CSV_PATH = (
    "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv"
)
PREDICTED_CSV = "predicted.csv"


def get_file_size(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def get_df_from_source(path, effective_train_data) -> (pd.DataFrame, int, int):
    min_epochs = 2
    line_count = get_file_size(path)

    epochs_ratio = effective_train_data / line_count

    if epochs_ratio < min_epochs:
        p = epochs_ratio / min_epochs
        epochs = min_epochs
    else:
        p = 1
        epochs = int(epochs_ratio)

    df = pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and np.random.random() > p,
    )
    df = df.fillna(0)
    df = df.query("date > 85").reset_index(drop=True)
    validation_size = int(df.shape[0] / 10)
    return df, epochs, validation_size


def strip_for_batch_size(df, batch_size):
    strip = 1 + df.shape[0] % batch_size
    return df.loc[: df.shape[0] - strip]


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


class JaneStreetDatasetPredict(Dataset):
    def __init__(self, df, device, transform=None, batch_size=None):
        df.insert(2, "trade", None)
        df.loc[df["resp"] <= 0, "trade"] = 0
        df.loc[df["resp"] > 0, "trade"] = 1
        for one in range(4):
            i = 4 - one
            resp_str = "resp_{}".format(i)
            trade_str = "trade_{}".format(i)
            df.insert(2, trade_str, None)
            df.loc[df[resp_str] <= 0, trade_str] = 0
            df.loc[df[resp_str] > 0, trade_str] = 1

        df.trade = df.trade.multiply(1)
        df.resp = df.resp.multiply(1)
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.batch_size = batch_size

    # Getter
    def __getitem__(self, index):
        x = (
            torch.tensor(self.df.iloc[index]["feature_0":"feature_129"])
            .float()
            .to(self.device)
        )
        y = torch.tensor(self.df.iloc[index]["resp_1":"resp"]).float().to(self.device)

        item = (x, y)
        if self.transform:
            item = self.transform(item)
        return item

    # Get Length
    def __len__(self):
        return self.len


def get_core_model(
    input_size,
    output_size,
    hidden_count,
    dropout_p=0.15,
    net_width=32,
):
    assert hidden_count > 0
    layers = []

    def append_layer(layers, _input_size, _output_size, just_linear=False):
        layers.append(torch.nn.Linear(_input_size, _output_size))
        if just_linear:
            return
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(torch.nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(_output_size))

    append_layer(layers, input_size, net_width)

    for one in range(hidden_count):
        append_layer(layers, net_width, net_width)

    append_layer(layers, net_width, output_size, just_linear=True)
    return torch.nn.Sequential(*layers)


class CustomSmoothL1Loss(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = torch.clone(target)
        # where we have missmatch in signs - we can make te loss bigger
        # by multiplying y by constant making distance between x an y bigger and therefore loss bigger
        target[input * target < 0] = target[input * target < 0] * 3
        # target[target == 10] = 20
        # target[target == -10] = -20
        ret = super().forward(input, target)
        return ret


def train_model_predict(
    model,
    train_loader,
    validation_loader,
    n_epoch=None,
    lr=None,
    phase=None,
):
    writer = SummaryWriter()
    criterion = nn.SmoothL1Loss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = CustomSmoothL1Loss()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = torch.optim.Adadelta(model.parameters())
    # optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.Adagrad(model.parameters())
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode="min", factor=0.1, patience=10
    # )
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr)
    LBFGS = True
    if not LBFGS:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=10, history_size=10, lr=0.2
        )
    accuracy_list = {}
    loss_list = {}
    scheduler_count = 0
    for epoch in range(n_epoch):
        for x, y in train_loader:
            scheduler_count += 1
            model.train()
            if not LBFGS:
                optimizer.zero_grad()
                z = model(x.float())
                loss = criterion(z, y.float())
                writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
                loss.backward()

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                z = model(x.float())
                loss = criterion(z, y.float())
                if loss.requires_grad:
                    loss.backward()
                return loss

            if LBFGS:
                optimizer.step(closure)
            else:
                optimizer.step()

            if LBFGS:
                z = model(x.float())
                loss = criterion(z, y.float())
                writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())

            if scheduler_count % 5000 == 0:
                print("500 batches passed")
                # scheduler.step(scheduler_count / 40000)

        # perform a prediction on the validation data
        accurate_guess = 0
        accurate_guess_from_random = 0
        accurate_guess_from_mean = 0
        accurate_guess_from_median = 0
        total_count = 1

        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())

            for one in range(len(z)):
                # if y_test[one][1] > 0:
                total_count += 1
                rand_choice = int(torch.randint(0, 2, (1, 1))[0])
                if rand_choice == 0:
                    rand_choice = -1
                if z[one][-1] * y_test[one][-1] > 0:
                    accurate_guess += 1
                if z[one][0] * rand_choice > 0:
                    accurate_guess_from_random += 1

                if torch.mean(z[one][1:]) * y_test[one][0] > 0:
                    accurate_guess_from_mean += 1

                if torch.median(z[one][1:]) * y_test[one][0] > 0:
                    accurate_guess_from_median += 1

            loss = criterion(z, y_test.float())
            writer.add_scalar("Validation data phase: {}".format(phase), loss, epoch)
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())
        print("epoch_{} accuracy: {}".format(epoch, accurate_guess / total_count))
        print(
            "epoch_{} accuracy from random choice: {}".format(
                epoch, accurate_guess_from_random / total_count
            )
        )
        print(
            "epoch_{} accuracy from resps mean: {}".format(
                epoch, accurate_guess_from_mean / total_count
            )
        )
        print(
            "epoch_{} accuracy from resps median: {}".format(
                epoch, accurate_guess_from_median / total_count
            )
        )

    model.eval()

    return (accuracy_list, loss_list)


def create_and_train_predict_model(
    df, validation_size=1000, batch_size=200, n_epoch=2, lr=0.05, device=None
):
    df = df.fillna(0)
    frac = 1 / 1
    print("n_epoch {}".format(n_epoch))

    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index()
    train_sample = strip_for_batch_size(train_sample, batch_size)
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index()
    validation_sample = strip_for_batch_size(validation_sample, batch_size)

    janestreet_train = JaneStreetDatasetPredict(df=train_sample, device=device)
    janestreet_validation = JaneStreetDatasetPredict(
        df=validation_sample, device=device
    )
    train_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet_train, batch_size=batch_size, drop_last=True
    )
    validation_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet_validation, batch_size=batch_size, drop_last=True
    )
    model = get_core_model(
        130,
        5,
        hidden_count=3,
        dropout_p=0.2,
        net_width=1000,
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    model = model.float()
    model.to(device)

    train_res = train_model_predict(
        model,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
        lr=lr,
        phase=1,
    )
    (accuracy_list, loss_list) = train_res

    print_losses(accuracy_list, n_epoch, 1)

    return model


def main(
    batch_size=2000,
    effective_train_data=400000,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, n_epoch, validation_size = get_df_from_source(CSV_PATH, effective_train_data)
    model = create_and_train_predict_model(
        df,
        validation_size=validation_size,
        batch_size=batch_size,
        n_epoch=n_epoch,
        lr=0.2,
        device=device,
    )
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
