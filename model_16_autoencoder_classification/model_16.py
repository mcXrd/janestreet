# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import Tensor
import fire

jane_street_submission = False

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

if jane_street_submission:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    CSV_PATH = "/kaggle/input/jane-street-market-prediction/train.csv"
else:
    CSV_PATH = (
        "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv"
    )
ENCODED_CSV = "encoded.csv"
USE_FINISHED_ENCODE = True
import os

try:
    if not USE_FINISHED_ENCODE:
        os.remove(ENCODED_CSV)
except FileNotFoundError:
    pass

# assert torch.cuda.is_available()
DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def get_file_size(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def get_df_from_source(path, effective_train_data) -> (pd.DataFrame, int, int):
    min_epochs = 2
    max_line_count = 2000000
    line_count = get_file_size(path)

    epochs_ratio = effective_train_data / line_count

    if epochs_ratio < min_epochs:
        p = epochs_ratio / min_epochs
        epochs = min_epochs
    else:
        p = 1
        epochs = int(epochs_ratio)

    if line_count * p > max_line_count:
        p = p * (max_line_count / line_count)

    df = pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and np.random.random() > p,
    )
    df = df.fillna(0)
    # df = df.query("date > 85").reset_index(drop=True)
    validation_size = int(df.shape[0] / 10)
    return df, epochs, validation_size


def strip_for_batch_size(df, batch_size):
    strip = 1 + df.shape[0] % batch_size
    return df.loc[: df.shape[0] - strip]


class AddNoiseMixin:
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.2)

    @staticmethod
    def reverse_mask(mask):
        mask = torch.clone(mask)
        mask[mask == 0] = 2
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        return mask

    def get_mask(self, inputs):
        """
        zeros must mean corrupted - ones must mean uncorrupted
        """
        mask = torch.ones_like(inputs)
        mask = self.dropout_model(mask)
        mask[mask != 0] = 1
        return mask


class AddDropoutNoiseMixin(AddNoiseMixin):
    def add_noise(self, inputs):
        mask = self.get_mask(inputs)
        # zeros means corrupted
        inputs = inputs * mask
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 16)
        inputs = inputs + noise
        return inputs, mask


class JaneStreetEncode1Dataset(AddDropoutNoiseMixin, Dataset):
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.2)

    # Constructor with defult values
    def __init__(self, df, device, transform=None):
        self.init_dropout()
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device

    # Getter
    def __getitem__(self, index):
        inputs, mask = self.add_noise(
            torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values)
        )
        sample = (
            inputs,
            torch.from_numpy(self.df.iloc[index]["feature_0":"feature_129"].values),
            mask,
        )
        sample = (
            sample[0].to(self.device),
            sample[1].to(self.device),
            sample[2].to(self.device),
        )
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


class autoencoder(nn.Module):
    def __init__(self, input_size, output_size, bottleneck):
        super(autoencoder, self).__init__()
        self.encoder = get_core_model(
            input_size, bottleneck, 1, dropout_p=0.16, net_width=1500
        )
        self.decoder = get_core_model(
            bottleneck, output_size, 1, dropout_p=0.16, net_width=1500
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


class EmphasizedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0
    ) -> None:
        super().__init__(size_average, reduce, reduction, beta)
        self.emphasize_ratio = 0.5
        assert self.emphasize_ratio < 1

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        target = torch.clone(target)
        distance = torch.abs(input - target)
        # mask : zeros = corrupted
        distance = distance * mask
        distance = distance.float()
        # distance is only set for uncorrupted scalars, these are scalars which loss we wanna neglect a bit
        # we will neglect it by setting the target to be closer to the input - for corrupted case, the distance
        # will be zero - so their value will not change
        target[input > target] = (
            target[input > target] + distance[input > target] * self.emphasize_ratio
        )
        target[input < target] = (
            target[input < target] - distance[input < target] * self.emphasize_ratio
        )
        ret = super().forward(input, target)
        return ret  # mask : zeros = corrupted


def train_model_encoder(
    model,
    train_loader,
    validation_loader,
    n_epoch=None,
    phase=None,
):
    writer = SummaryWriter()
    criterion = EmphasizedSmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        for x, y, mask in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y.float(), mask)
            if not jane_street_submission:
                writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
            loss.backward()
            optimizer.step()
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())
        # perform a prediction on the validation data
        for x_test, y_test, mask_test in validation_loader:
            model.eval()
            z = model(x_test.float())
            loss = criterion(z, y_test.float(), mask_test)
            if not jane_street_submission:
                writer.add_scalar(
                    "Validation data phase: {}".format(phase), loss, epoch
                )
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())

    model.eval()

    # get encoded layer

    def build_encoded_layer(loader):
        encoded_layer = []
        for x, y, mask in loader:
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


def print_losses(accuracy_list, n_epoch, phase):
    print("phase {}".format(phase))
    print(
        "validation loss mean: {} {} {}".format(
            np.mean(accuracy_list[0]),
            np.mean(accuracy_list[-1]),
        )
    )
    print(
        "validation loss variance: {} {} {}".format(
            np.var(accuracy_list[0]),
            np.var(accuracy_list[-1]),
        )
    )


def extract_model_input(
    df, batch_size, device, encoder_model, encoded_features_count=50
):
    base_culumns = ["date", "weight", "resp_1", "resp_2", "resp_3", "resp_4", "resp"]
    encoded_columns = []
    original_columns = []

    for one in range(130):
        original_columns.append("feature_{}".format(one))
    original_columns.append("ts_id")

    for one in range(encoded_features_count):
        encoded_columns.append("enc_feature_{}".format(one))

    batch = []
    batch_original_rows = []

    new_data = []
    new_df = pd.DataFrame(
        new_data,
        columns=base_culumns + original_columns + encoded_columns,
    )
    new_df.to_csv(ENCODED_CSV)

    def save_part_to_csv(_new_df, new_data):
        _new_df = pd.DataFrame(
            new_data,
            columns=base_culumns + original_columns + encoded_columns,
        )
        _new_df.to_csv(ENCODED_CSV, mode="a", header=False)

    for index, row in df.iterrows():
        if len(new_data) > 10000:
            save_part_to_csv(new_df, new_data)
            new_data = []

        a = row["feature_0":"feature_129"].values
        batch.append(a)
        batch_original_rows.append(row)

        if len(batch) == batch_size:

            model_input = torch.from_numpy(np.array(batch)).float().to(device)
            z = encoder_model.encoder(model_input)
            assert len(z) == len(batch_original_rows)
            for i in range(len(z)):
                row_list = (
                    batch_original_rows[i].tolist()
                    + z[i].cpu().detach().numpy().tolist()
                )
                new_data.append(row_list)

            batch = []
            batch_original_rows = []
    save_part_to_csv(new_df, new_data)
    new_data = []

    new_df = pd.read_csv(
        ENCODED_CSV,
        nrows=1000,
    )
    pd.testing.assert_frame_equal(
        new_df.loc[0:899, "date":"feature_129"],
        df.loc[0:899, "date":"feature_129"],
        check_dtype=False,
    )

    test_new_df_source = new_df.loc[200 : 199 + batch_size, "feature_0":"feature_129"]
    test_new_df_encode = new_df.loc[
        200 : 199 + batch_size, "enc_feature_0":"enc_feature_49"
    ]
    model_input = torch.from_numpy(np.array(test_new_df_source)).float().to(device)
    z = encoder_model.encoder(model_input)
    test_new_df_fresh_encode = pd.DataFrame(
        z.cpu().detach().numpy(),
        columns=encoded_columns,
    )
    pd.testing.assert_frame_equal(
        test_new_df_encode.loc[
            200 : 199 + batch_size, "enc_feature_0":"enc_feature_49"
        ].reset_index(drop=True),
        test_new_df_fresh_encode.loc[:, "enc_feature_0":"enc_feature_49"].reset_index(
            drop=True
        ),
        check_dtype=False,
    )


def create_autencoder_and_train(
    batch_size=200,
    effective_train_data=1000000,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, n_epoch, validation_size = get_df_from_source(CSV_PATH, effective_train_data)
    print("n_epoch {}".format(n_epoch))
    frac = 1 / 1
    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index(drop=True)
    train_sample = strip_for_batch_size(train_sample, batch_size)
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index(drop=True)
    validation_sample = strip_for_batch_size(validation_sample, batch_size)

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
        input_size=130,
        output_size=130,
        bottleneck=50,
    )
    model1 = model1.float()
    model1.to(device)

    train_res = train_model_encoder(
        model1,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
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

    return model1


class JaneStreetDatasetPredict(Dataset):

    # Constructor with defult values
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

        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.batch_size = batch_size

    # Getter
    def __getitem__(self, index):
        x = (
            torch.tensor(self.df.iloc[index]["enc_feature_0":"enc_feature_49"])
            .float()
            .to(self.device)
        )
        y = torch.tensor(self.df.iloc[index]["trade_1":"trade"]).float().to(self.device)

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
    dropout_p=0.16,
    net_width=32,
):
    assert hidden_count > 0
    layers = []

    def append_layer(layers, _input_size, _output_size, just_linear=False):
        layers.append(torch.nn.Linear(_input_size, _output_size))
        if just_linear:
            return
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(torch.nn.ELU())
        layers.append(nn.BatchNorm1d(_output_size))

    append_layer(layers, input_size, net_width)

    for one in range(hidden_count):
        append_layer(layers, net_width, net_width)

    append_layer(layers, net_width, output_size, just_linear=True)
    return torch.nn.Sequential(*layers)


def train_model_predict(
    model,
    train_loader,
    validation_loader,
    n_epoch=None,
    lr=None,
    phase=None,
):
    writer = SummaryWriter()
    criterion = nn.BCEWithLogitsLoss()
    LBFGS = True
    if not LBFGS:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    else:
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=2, history_size=10, lr=0.2
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
                if not jane_street_submission:
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
                if not jane_street_submission:
                    writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())

        # perform a prediction on the validation data
        accurate_guess = 0
        accurate_guess_from_random = 0
        accurate_guess_from_mean = 0
        accurate_guess_from_median = 0
        total_count = 0

        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())

            for one in range(len(z)):
                # if y_test[one][1] > 0:
                total_count += 1
                rand_choice = int(torch.randint(0, 2, (1, 1))[0])
                if rand_choice == 0:
                    rand_choice = -1
                if (
                    torch.sigmoid_(z[one][-1]) > 0.5
                    and y_test[one][-1] == 1
                    or torch.sigmoid_(z[one][-1]) <= 0.5
                    and y_test[one][-1] == 0
                ):
                    accurate_guess += 1
                if z[one][-1] * rand_choice > 0:
                    accurate_guess_from_random += 1

                if (
                    torch.mean(torch.sigmoid_(z[one])) > 0.5
                    and y_test[one][-1] == 1
                    or torch.mean(torch.sigmoid_(z[one])) <= 0.5
                    and y_test[one][-1] == 0
                ):
                    accurate_guess_from_mean += 1
                if (
                    torch.median(torch.sigmoid_(z[one])) > 0.5
                    and y_test[one][-1] == 1
                    or torch.median(torch.sigmoid_(z[one])) <= 0.5
                    and y_test[one][-1] == 0
                ):
                    accurate_guess_from_median += 1

            loss = criterion(z, y_test.float())
            if not jane_street_submission:
                writer.add_scalar(
                    "Validation data phase: {}".format(phase), loss, epoch
                )
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


def batchify(row, batch_size):
    row = list(row)
    ret = []
    for one in range(batch_size):
        ret.append(row)
    ret = np.array(ret)
    return ret


def model_input_from_row(row, batch_size, device):
    batch_of_rows = batchify(row, batch_size)
    model_input = torch.from_numpy(batch_of_rows).float().to(device)
    return model_input


def visual_control_predict(model, batch_size, device, df):
    model.eval()
    index = np.random.randint(1, len(df))
    row = df.iloc[index]["enc_feature_0":"enc_feature_49"].values
    model_input = model_input_from_row(row, batch_size, device)
    z = model(model_input)
    print("Model output")
    print(z[0])
    print("actual_data")
    print(df.iloc[index]["trade":"resp"])


def create_and_train_predict_model(
    df,
    validation_size=1000,
    batch_size=200,
    n_epoch=2,
    lr=0.05,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = df.fillna(0)
    frac = 1 / 1
    print("n_epoch {}".format(n_epoch))

    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index(drop=True)
    train_sample = strip_for_batch_size(train_sample, batch_size)
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index(drop=True)
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
        50,
        5,
        hidden_count=6,
        dropout_p=0.15,
        net_width=150,
    )

    model = model.float()
    model.to(device)

    train_res = train_model_predict(
        model,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
        lr=lr,
        phase=4,
    )
    (accuracy_list, loss_list) = train_res

    print_losses(accuracy_list, n_epoch, 1)

    visual_control_predict(
        model,
        batch_size,
        device,
        validation_sample,
    )
    return model


def main(
    batch_size=500,
    effective_train_data=1000000,
):
    if not USE_FINISHED_ENCODE:
        encoder_model = create_autencoder_and_train(
            batch_size=batch_size,
            effective_train_data=effective_train_data,
        )

        df, n_epoch, validation_size = get_df_from_source(
            CSV_PATH, effective_train_data
        )
        df = strip_for_batch_size(df, batch_size)
        extract_model_input(
            df,
            batch_size,
            DEVICE,
            encoder_model,
            encoded_features_count=50,
        )
    df, n_epoch, validation_size = get_df_from_source(ENCODED_CSV, effective_train_data)

    predict_model = create_and_train_predict_model(
        df,
        validation_size=validation_size,
        batch_size=batch_size,
        n_epoch=n_epoch,
        lr=0.05,
    )
    print("Done")
    if jane_street_submission:
        return encoder_model, predict_model


if not jane_street_submission and __name__ == "__main__":
    fire.Fire(main)

if jane_street_submission:
    import janestreet

    BATCH_SIZE = 500
    encoder_model, predict_model = main(
        effective_train_data=2000000, batch_size=BATCH_SIZE
    )
    print("models ready")
    env = janestreet.make_env()
    for (test_df, pred_df) in tqdm(env.iter_test()):
        if not test_df["weight"].item() > 0:
            pred_df.action = 0
            env.predict(pred_df)
            continue
        row = test_df.loc[:, "feature_0":"feature_129"]
        row = row.iloc[0]
        row = model_input_from_row(row.values, BATCH_SIZE, DEVICE)
        z = predict_model(encoder_model.encoder(row))

        pred_df.action = 1 if torch.sigmoid_(z[0][-1]) > 0.5 else 0
        env.predict(pred_df)

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
