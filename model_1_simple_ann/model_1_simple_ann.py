import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire


def main(
    validation_size=1000,
    nrows=10000,
    dropout_p=0.0,
    batch_size=100,
    n_epoch=10,
    validation_results_count=20,
    big_number=100,
    small_number=20,
):
    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    df = pd.read_csv(
        "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv",
        nrows=nrows,
    )
    df = df.fillna(0)
    df["big_resp"] = df["resp"] * 10000

    class JaneStreetDataset(Dataset):

        # Constructor with defult values
        def __init__(self, df, transform=None):
            self.df = df
            self.len = len(self.df)
            self.transform = transform

        # Getter
        def __getitem__(self, index):
            sample = (
                torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values),
                torch.tensor(self.df.iloc[index]["big_resp"]),
            )
            sample = sample[0].to(device), sample[1].to(device)
            if self.transform:
                sample = self.transform(sample)
            return sample

        # Get Length
        def __len__(self):
            return self.len

    model = torch.nn.Sequential(
        torch.nn.Linear(130, small_number),
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
        torch.nn.Linear(big_number, big_number),
        nn.Dropout(p=dropout_p),
        torch.nn.ReLU(),
        nn.BatchNorm1d(big_number),
        torch.nn.Linear(big_number, small_number),
        nn.Dropout(p=dropout_p),
        torch.nn.ReLU(),
        nn.BatchNorm1d(small_number),
        torch.nn.Linear(small_number, 1),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    janestreet_train = JaneStreetDataset(df=df.iloc[:-validation_size])
    janestreet_validation = JaneStreetDataset(df=df.iloc[-validation_size:])
    train_loader = torch.utils.data.DataLoader(
        dataset=janestreet_train, batch_size=batch_size, drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=janestreet_validation, batch_size=batch_size, drop_last=True
    )

    model = model.float()
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.001)
    criterion = nn.MSELoss()

    def train_model(
        model, train_loader, validation_loader, optimizer, criterion, n_epoch=None
    ):
        accuracy_list = {}
        loss_list = {}
        for epoch in range(n_epoch):
            for x, y in train_loader:
                model.train()
                optimizer.zero_grad()
                z = model(x.float())
                y = y.view(-1, 1).float()
                loss = criterion(z, y)
                writer.add_scalar("Train data", loss, epoch)
                loss.backward()
                optimizer.step()
                if not loss_list.get(epoch):
                    loss_list[epoch] = []
                loss_list[epoch].append(loss.data)
            # perform a prediction on the validation data
            for x_test, y_test in validation_loader:
                model.eval()
                z = model(x_test.float())
                y = y.view(-1, 1)
                loss = criterion(z, y.float())
                writer.add_scalar("Validation data", loss, epoch)
                if not accuracy_list.get(epoch):
                    accuracy_list[epoch] = []
                accuracy_list[epoch].append(loss.data)
        return accuracy_list, loss_list

    accuracy_list, loss_list = train_model(
        model, train_loader, validation_loader, optimizer, criterion, n_epoch=n_epoch
    )
    print(accuracy_list[n_epoch-1])

    model.eval()

    for one in range(validation_results_count):
        index = np.random.randint(nrows - validation_size + 1, nrows - 1)
        # a = torch.from_numpy(df.iloc[index]["feature_0":"feature_129"].values)
        a = df.iloc[index]["feature_0":"feature_129"].values
        resp = torch.tensor(df.iloc[index]["big_resp"])

        r = []
        for one in range(batch_size):
            r.append(a)
        r = np.array(r)
        model_input = torch.from_numpy(r).float().to(device)
        z = model(model_input)

        print(float(resp), float(z[0]))


if __name__ == "__main__":
    fire.Fire(main)
