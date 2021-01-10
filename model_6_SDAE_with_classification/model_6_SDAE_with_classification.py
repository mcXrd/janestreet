import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire

CSV_PATH = (
    "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv"
)


class JaneStreetDataset(Dataset):

    # Constructor with defult values
    def __init__(
        self, df, device, transform=None, e1=None, e2=None, e3=None, batch_size=None
    ):
        df.loc[df["resp"] <= 0, "trade"] = 0
        df.loc[df["resp"] > 0, "trade"] = 1
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.batch_size = batch_size

    def extract_model_input(self, index):
        a = self.df.iloc[index]["feature_0":"feature_129"].values
        r = []
        for one in range(self.batch_size):
            r.append(a)
        r = np.array(r)
        model_input = torch.from_numpy(r).float().to(self.device)
        z = self.e3(self.e2(self.e1(model_input)))
        return z[0]

    # Getter
    def __getitem__(self, index):
        torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values)
        sample = (
            self.extract_model_input(index),
            torch.tensor(self.df.iloc[index]["trade"]).float().to(self.device),
            # torch.from_numpy(self.df.iloc[index]["trade"]),
        )
        # sample = sample[0].to(self.device), sample[1].to(self.device)
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


class JaneStreetEncode1Dataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None):
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        # noise = inputs.clone().uniform_(-1, 1)
        noise = noise.multiply(1 / 16)
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
    def __init__(
        self,
        encoded_layer,
        device,
        transform=None,
        noise_type=None,
        noise_multiplicator=None,
        noise_dropout=None,
    ):
        self.encoded_layer = encoded_layer
        self.len = len(encoded_layer)
        self.transform = transform
        self.device = device
        self.noise_type = noise_type
        self.noise_multiplicator = noise_multiplicator
        self.noise_dropout = noise_dropout

    def add_noise(self, inputs):
        assert self.noise_multiplicator
        assert self.noise_dropout is not None
        dropout_model = nn.Dropout(p=self.noise_dropout)
        if self.noise_type == "G":
            noise = torch.randn_like(inputs)
        elif self.noise_type == "U":
            noise = inputs.clone().uniform_()
        elif self.noise_type == "A":
            multiplier = 1 + np.random.randint(-3, 3) / 100  # from 0.97 to 1.03
            inputs = inputs.multiply(multiplier)
            return inputs
        else:
            raise Exception("Unknown noise")
        noise = dropout_model(noise)
        noise = noise.multiply(self.noise_multiplicator)
        return inputs + noise

    # Getter
    def __getitem__(self, index):
        sample = (
            self.add_noise(torch.from_numpy(self.encoded_layer[index])),
            torch.from_numpy(self.encoded_layer[index]),
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


class classifier(nn.Module):
    def __init__(
        self,
        small_number,
        big_number,
        dropout_p,
    ):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(55, small_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(small_number),
            torch.nn.Linear(small_number, big_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(big_number),
            torch.nn.Linear(big_number, big_number),
            nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            nn.BatchNorm1d(big_number),
            torch.nn.Linear(big_number, 1),
        )

    def forward(self, x):
        x = self.m(x)
        return x


def train_model_classification(
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
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    if criterion is None:
        criterion = nn.SmoothL1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr / 3)
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
    return (
        accuracy_list,
        loss_list,
        None,
        None,
    )


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
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    if criterion is None:
        criterion = nn.SmoothL1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr / 3)
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


def visual_control(
    model, nrows, validation_size, batch_size, device, df, validation_encoded_layer
):
    model.eval()
    index = np.random.randint(0, len(validation_encoded_layer))
    actual_data = df.iloc[index]["feature_0":"feature_129"].values
    a = validation_encoded_layer[index]
    r = []
    for one in range(batch_size):
        r.append(a)
    r = np.array(r)
    model_input = torch.from_numpy(r).float().to(device)
    z = model(model_input)
    print("Input data")
    print("len {}".format(len(a)))
    print(a[:20])
    print("actual_data")
    print("len {}".format(len(actual_data)))
    print(actual_data[:20])
    print("Denoised data")
    print("len {}".format(len(z[0])))
    print(z[0][:20])


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


def main(
    nrows=400000,
    big_number=500,
    small_number=500,
    dropout_p=0.15,
    validation_size=40000,
    batch_size=200,
    n_epoch=25,
    lr=0.15,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(
        CSV_PATH,
        nrows=nrows,
        # skiprows=range(1, 500000),
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
    (
        accuracy_list,
        loss_list,
        train_encoded_layer1,
        validation_encoded_layer1,
    ) = train_res
    assert len(train_encoded_layer1) == len(janestreet1_train)
    assert len(validation_encoded_layer1) == len(janestreet1_validation)

    print_losses(accuracy_list, n_epoch, 1)

    janestreet2_train = JaneStreetEncode2Dataset(
        encoded_layer=train_encoded_layer1,
        device=device,
        noise_type="U",
        noise_multiplicator=1 / 16,
        noise_dropout=0.0,
    )
    janestreet2_validation = JaneStreetEncode2Dataset(
        encoded_layer=validation_encoded_layer1,
        device=device,
        noise_type="U",
        noise_multiplicator=1 / 16,
        noise_dropout=0.0,
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
        output_size=60,
        bottleneck=55,
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

    print_losses(accuracy_list, n_epoch, 2)

    janestreet3_train = JaneStreetEncode2Dataset(
        encoded_layer=train_encoded_layer2,
        device=device,
        noise_type="A",
        noise_multiplicator=1 / 32,
        noise_dropout=0.0,
    )
    janestreet3_validation = JaneStreetEncode2Dataset(
        encoded_layer=validation_encoded_layer2,
        device=device,
        noise_type="A",
        noise_multiplicator=1 / 32,
        noise_dropout=0.0,
    )
    train_loader3 = torch.utils.data.DataLoader(
        dataset=janestreet3_train, batch_size=batch_size, drop_last=True
    )
    validation_loader3 = torch.utils.data.DataLoader(
        dataset=janestreet3_validation, batch_size=batch_size, drop_last=True
    )

    model3 = autoencoder(
        small_number=small_number,
        big_number=big_number,
        dropout_p=dropout_p,
        input_size=55,
        output_size=55,
        bottleneck=50,
    )
    model3 = model3.float()
    model3.to(device)

    train_res = train_model(
        model3,
        train_loader3,
        validation_loader3,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
        phase=3,
    )
    (
        accuracy_list,
        loss_list,
        train_encoded_layer3,
        validation_encoded_layer3,
    ) = train_res

    assert len(train_encoded_layer3) == len(janestreet3_train)
    assert len(validation_encoded_layer3) == len(janestreet3_validation)

    print_losses(accuracy_list, n_epoch, 3)

    janestreet_classifier_train = JaneStreetDataset(
        df=train_sample,
        device=device,
        e1=model1,
        e2=model2,
        e3=model3,
        batch_size=batch_size,
    )
    janestreet_classifier_validation = JaneStreetDataset(
        df=validation_sample,
        device=device,
        e1=model1,
        e2=model2,
        e3=model3,
        batch_size=batch_size,
    )
    train_classifier_loader = torch.utils.data.DataLoader(
        dataset=janestreet_classifier_train, batch_size=batch_size, drop_last=True
    )
    validation_classifier_loader = torch.utils.data.DataLoader(
        dataset=janestreet_classifier_validation, batch_size=batch_size, drop_last=True
    )

    model_classifier = classifier(
        small_number=small_number, big_number=big_number, dropout_p=dropout_p
    )

    model_classifier = model_classifier.float()
    model_classifier.to(device)

    criterion = nn.BCEWithLogitsLoss()
    train_res = train_model_classification(
        model_classifier,
        train_classifier_loader,
        validation_classifier_loader,
        n_epoch=n_epoch,
        lr=lr,
        device=device,
        phase=4,
        criterion=criterion,
    )
    (
        accuracy_list,
        loss_list,
        train_encoded_layer3,
        validation_encoded_layer3,
    ) = train_res

    print_losses(accuracy_list, n_epoch, 4)

    visual_control(
        model_classifier,
        nrows,
        validation_size,
        batch_size,
        device,
        validation_sample,
        validation_encoded_layer2,
    )

    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
