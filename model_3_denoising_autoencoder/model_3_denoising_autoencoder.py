import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire


class autoencoder(nn.Module):
    def __init__(self, small_number, big_number, dropout_p, input_size, bottleneck):
        super(autoencoder, self).__init__()
        bottleneck = bottleneck
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, small_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(small_number),
            torch.nn.Linear(small_number, big_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(big_number),
            torch.nn.Linear(small_number, bottleneck),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck, small_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(small_number),
            torch.nn.Linear(small_number, big_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(big_number),
            torch.nn.Linear(small_number, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


class JaneStreetDataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None):
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        print(self.device)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1/8)
        return inputs + noise

    # Getter
    def __getitem__(self, index):
        sample = (
            self.add_noise(
                torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values)
            ),
            torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values),
        )
        sample = sample[0].to(self.device), sample[1].to(self.device)
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


def train_model(
    model, train_loader, validation_loader, n_epoch=None, lr=None, device=None
):
    writer = SummaryWriter()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr)
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            # y = y.view(-1, 1).float()
            # y = y.float()
            loss = criterion(z, y.float())
            writer.add_scalar("Train data", loss, epoch)
            loss.backward()
            optimizer.step()
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())
        # perform a prediction on the validation data
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())
            # z = torch.zeros((100, 130)).to(device)
            # z = y_test
            # y_test = y_test.view(-1, 1)
            # y_test = y_test.float()
            loss = criterion(z, y_test.float())
            writer.add_scalar("Validation data", loss, epoch)
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())
    return accuracy_list, loss_list


def visual_control(model, nrows, validation_size, batch_size, device, df):
    model.eval()
    index = np.random.randint(nrows - validation_size + 1, nrows - 1)
    a = df.iloc[index]["feature_0":"feature_129"].values
    r = []
    for one in range(batch_size):
        r.append(a)
    r = np.array(r)
    model_input = torch.from_numpy(r).float().to(device)
    z = model(model_input)
    print("Input data")
    print(a)
    print("Denoised data")
    print(z[0])


def main(
    nrows=400000,
    big_number=500,
    small_number=500,
    dropout_p=0.15,
    validation_size=40000,
    batch_size=200,
    n_epoch=100,
    lr=0.15,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = autoencoder(
        small_number=small_number,
        big_number=big_number,
        dropout_p=dropout_p,
    )
    model = model.float()
    model.to(device)

    df = pd.read_csv(
        "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv",
        nrows=nrows,
        skiprows=range(1, 500000),
    )
    df = df.fillna(0)
    df = df.multiply(1)
    frac = 1 / 20
    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index()
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index()
    nrows = int(nrows * frac)
    validation_size = int(validation_size * frac)

    janestreet_train = JaneStreetDataset(df=train_sample, device=device)
    janestreet_validation = JaneStreetDataset(df=validation_sample, device=device)
    train_loader = torch.utils.data.DataLoader(
        dataset=janestreet_train, batch_size=batch_size, drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=janestreet_validation, batch_size=batch_size, drop_last=True
    )

    accuracy_list, loss_list = train_model(
        model,
        train_loader,
        validation_loader,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
    )
    visual_control(model, nrows, validation_size, batch_size, device, df)

    print("start")
    print("validation loss mean: {}".format(np.mean(accuracy_list[1])))
    print("validation loss variance: {}".format(np.var(accuracy_list[1])))

    print("mid")
    print("validation loss mean: {}".format(np.mean(accuracy_list[int(n_epoch / 2)])))
    print(
        "validation loss variance: {}".format(np.var(accuracy_list[int(n_epoch / 2)]))
    )

    print("end")
    print("validation loss mean: {}".format(np.mean(accuracy_list[n_epoch - 1])))
    print("validation loss variance: {}".format(np.var(accuracy_list[n_epoch - 1])))


if __name__ == "__main__":
    fire.Fire(main)
