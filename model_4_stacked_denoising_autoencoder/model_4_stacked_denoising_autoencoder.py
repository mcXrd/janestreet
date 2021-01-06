import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire


class JaneStreetEncode1Dataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None):
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        print(self.device)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 8)
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


class JaneStreetEncode2Dataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, encoded_layer, transform=None):
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.encoded_layer = encoded_layer
        print(self.device)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 8)
        return inputs + noise

    # Getter
    def __getitem__(self, index):
        sample = (
            self.add_noise(torch.from_numpy(self.encoded_layer[index])),
            torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values),
        )
        sample = sample[0].to(self.device), sample[1].to(self.device)
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


class JaneStreetEncode3Dataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None):
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        print(self.device)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 8)
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


class autoencoder(nn.Module):
    def __init__(
        self, small_number, big_number, dropout_p, input_size, output_size, bottleneck
    ):
        super(autoencoder, self).__init__()
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
            torch.nn.Linear(small_number, output_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


def train_model(
    model,
    train_loader,
    validation_loader,
    n_epoch=None,
    lr=None,
    device=None,
    phase=None,
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
            writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
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
            writer.add_scalar("Validation data phase: {}".format(phase), loss, epoch)
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())

    model.eval()
    # get encoded layer

    def build_encoded_layer(loader):
        encoded_layer = []
        for x, y in loader:
            z = model.encoder(x.float())
            for row in z.cpu().detach().numpy():
                encoded_layer.append(row)
        return np.array(encoded_layer)

    return (
        accuracy_list,
        loss_list,
        build_encoded_layer(train_loader),
        build_encoded_layer(validation_loader),
    )


def main(
    nrows=4000,
    big_number=500,
    small_number=500,
    dropout_p=0.15,
    validation_size=40,
    batch_size=20,
    n_epoch=2,
    lr=0.15,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(
        "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv",
        nrows=nrows,
        skiprows=range(1, 500000),
    )
    df = df.fillna(0)
    df = df.multiply(1)
    frac = 1 / 1
    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index()
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index()
    nrows = int(nrows * frac)
    validation_size = int(validation_size * frac)

    janestreet1_train = JaneStreetEncode1Dataset(df=train_sample, device=device)
    janestreet1_validation = JaneStreetEncode1Dataset(
        df=validation_sample, device=device
    )
    train_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet1_train, batch_size=batch_size, drop_last=True
    )
    validation_loader1 = torch.utils.data.DataLoader(
        dataset=janestreet1_validation, batch_size=batch_size, drop_last=True
    )
    model1 = autoencoder(
        small_number=small_number,
        big_number=big_number,
        dropout_p=dropout_p,
        input_size=130,
        output_size=130,
        bottleneck=60,
    )
    model1 = model1.float()
    model1.to(device)

    train_res = train_model(
        model1,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
        phase=1,
    )
    accuracy_list, loss_list, train_encoded_layer, validation_encoded_layer = train_res
    assert len(train_encoded_layer) == len(janestreet1_train)
    assert len(validation_encoded_layer) == len(janestreet1_validation)
    print(accuracy_list)

    janestreet2_train = JaneStreetEncode2Dataset(
        df=train_sample, device=device, encoded_layer=train_encoded_layer
    )
    janestreet2_validation = JaneStreetEncode2Dataset(
        df=validation_sample, device=device, encoded_layer=validation_encoded_layer
    )
    train_loader2 = torch.utils.data.DataLoader(
        dataset=janestreet2_train, batch_size=batch_size, drop_last=True
    )
    validation_loader2 = torch.utils.data.DataLoader(
        dataset=janestreet2_validation, batch_size=batch_size, drop_last=True
    )
    model2 = autoencoder(
        small_number=small_number,
        big_number=big_number,
        dropout_p=dropout_p,
        input_size=60,
        output_size=130,
        bottleneck=60,
    )
    model2 = model2.float()
    model2.to(device)
    train_res = train_model(
        model2,
        train_loader2,
        validation_loader2,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
        phase=2,
    )
    (
        accuracy_list,
        loss_list,
        train_encoded_layer2,
        validation_encoded_layer2,
    ) = train_res

    assert len(train_encoded_layer2) == len(janestreet2_train)
    assert len(validation_encoded_layer2) == len(janestreet2_validation)

    print("DONE")


if __name__ == "__main__":
    fire.Fire(main)
