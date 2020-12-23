import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import fire


def main(nrows=10000, big_number=300, small_number=60, dropout_p=0.15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class JaneStreetDatasetDenoisingAutoencoder(Dataset):

        # Constructor with defult values
        def __init__(self, transform=None):
            df = pd.read_csv(
                "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv",
                nrows=nrows,
                # skiprows=skiprows,
            )
            df = df.fillna(0)
            self.df = df
            self.len = len(self.df)
            self.transform = transform

        def add_noise(self, inputs):
            noise = torch.randn_like(inputs)
            return inputs + noise

        # Getter
        def __getitem__(self, index):
            sample = (
                self.add_noise(
                    torch.from_numpy(
                        self.df.iloc[index]["feature_0":"feature_129"].values
                    )
                ),
                torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values),
            )
            sample = sample[0].to(device), sample[1].to(device)
            if self.transform:
                sample = self.transform(sample)
            return sample

        # Get Length
        def __len__(self):
            return self.len

    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(130, small_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(small_number),
                torch.nn.Linear(small_number, big_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(big_number),
                torch.nn.Linear(big_number, small_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(small_number),
                torch.nn.Linear(small_number, 130),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(130, small_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(small_number),
                torch.nn.Linear(small_number, big_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(big_number),
                torch.nn.Linear(big_number, small_number),
                nn.Dropout(p=dropout_p),
                torch.nn.ReLU(),
                nn.BatchNorm1d(small_number),
                torch.nn.Linear(small_number, 130),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


if __name__ == "__main__":
    fire.Fire(main)
