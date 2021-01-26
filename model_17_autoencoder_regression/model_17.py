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
from typing import Tuple, Callable, List

import os

JANE_STREET_SUBMISSION = False

if not JANE_STREET_SUBMISSION:
    import fire

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

if JANE_STREET_SUBMISSION:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    CSV_PATH = "/kaggle/input/jane-street-market-prediction/train.csv"
else:
    CSV_PATH = (
        "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/train.csv"
    )
ENCODED_CSV = "encoded.csv"
USE_FINISHED_ENCODE = False
LBFGS = False
FLOAT_SIZE = "float64"
WEIGHTED_RESPS = True
DATE_OVERFIT_FILL = None

try:
    if not USE_FINISHED_ENCODE:
        os.remove(ENCODED_CSV)
except FileNotFoundError:
    pass

# assert torch.cuda.is_available()
DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
ENCODED_FEATURES_COUNT = 70


def get_file_size(filename: str) -> int:
    with open(filename) as f:
        return sum(1 for line in f)


def add_time_columns_to_df(df: pd.DataFrame, eval: bool = False) -> pd.DataFrame:
    # "w_ts_id" 0 means beginning of the day, 1 means end of the day
    # "w_date" 0 means first day, 1 means last day
    # 0.5 is middle day/dataset
    feature_0_pos = df.columns.get_loc("feature_0")
    df.insert(feature_0_pos, "w_date", None)
    if DATE_OVERFIT_FILL is None:
        df.loc[:, "w_date"] = 1
    if eval and DATE_OVERFIT_FILL:
        df.loc[:, "w_date"] = DATE_OVERFIT_FILL
    else:
        df.loc[:, "w_date"] = df["date"] / df["date"].max()

    weight_w_date = 10 if DATE_OVERFIT_FILL else 1
    df.loc[:, "w_date"] = df.loc[:, "w_date"] * weight_w_date
    df = df.astype(FLOAT_SIZE)
    return df


def get_df_from_source(
    path: str, effective_train_data: int
) -> (pd.DataFrame, int, int):
    max_line_count = 1000000
    line_count = get_file_size(path)
    print("line count: {}".format(line_count))

    p = 1
    if line_count > max_line_count:
        p = max_line_count / line_count

    if effective_train_data > line_count:
        epochs = int(effective_train_data / line_count)
    if effective_train_data <= line_count:
        epochs = 1
        p = effective_train_data / line_count

    df = pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and np.random.random() > p,
    )
    df = df.fillna(0)
    df = df.astype(FLOAT_SIZE)
    # df = df.query("date > 85").reset_index(drop=True)
    validation_size = int(df.shape[0] / 10)
    print(df.shape)
    if WEIGHTED_RESPS:
        df = df[df["weight"] != 0]
        df = df.reset_index(drop=True)
        df["resp_1"] = df["resp_1"] * df["weight"]
        df["resp_2"] = df["resp_2"] * df["weight"]
        df["resp_3"] = df["resp_3"] * df["weight"]
        df["resp_4"] = df["resp_4"] * df["weight"]
        df["resp"] = df["resp"] * df["weight"]

    return df, epochs, validation_size


def strip_for_batch_size(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    strip = 1 + df.shape[0] % batch_size
    return df.loc[: df.shape[0] - strip]


class AddNoiseMixin:
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.2)

    @staticmethod
    def reverse_mask(mask: Tensor) -> Tensor:
        mask = torch.clone(mask)
        mask[mask == 0] = 2
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        return mask

    def get_mask(self, inputs: Tensor) -> Tensor:
        """
        zeros must mean corrupted - ones must mean uncorrupted
        """
        mask = torch.ones_like(inputs)
        mask = self.dropout_model(mask)
        mask[mask != 0] = 1
        return mask


class AddDropoutNoiseMixin(AddNoiseMixin):
    def add_noise(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        mask = self.get_mask(inputs)
        # zeros means corrupted
        inputs = inputs * mask
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 16)
        inputs = inputs + noise
        return inputs, mask


class JaneStreetEncode1Dataset(AddDropoutNoiseMixin, Dataset):
    Y_START_COLUMN = "w_date"
    Y_END_COLUMN = "feature_129"

    Y_LEN = 131

    @staticmethod
    def get_y_from_df(df: pd.DataFrame, index: int) -> Tensor:
        y = torch.from_numpy(
            df.iloc[index][
                JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN
            ].values
        )
        return y

    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.2)

    # Constructor with defult values
    def __init__(
        self, df: pd.DataFrame, device: torch.DeviceObjType, transform: Callable = None
    ):
        self.init_dropout()
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device

    # Getter
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        y = JaneStreetEncode1Dataset.get_y_from_df(self.df, index)
        assert len(y) == JaneStreetEncode1Dataset.Y_LEN
        inputs, mask = self.add_noise(y)
        sample = (
            inputs,
            y,
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
    def __init__(self, input_size: int, output_size: int, bottleneck: int):
        super(autoencoder, self).__init__()
        self.encoder = get_core_model(
            input_size, bottleneck, 1, dropout_p=0.16, net_width=1500
        )
        self.decoder = get_core_model(
            bottleneck, output_size, 1, dropout_p=0.16, net_width=1500
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = torch.sigmoid_(x)
        x = self.decoder(x)
        return x


class EmphasizedSmoothL1Loss(nn.SmoothL1Loss):
    # Autoencoders that don’t overfit towards the Identity
    # https://proceedings.neurips.cc/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf
    # and
    # Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion
    # https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
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
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    n_epoch: int = None,
    phase: int = None,
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
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
            if not JANE_STREET_SUBMISSION:
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
            if not JANE_STREET_SUBMISSION:
                writer.add_scalar(
                    "Validation data phase: {}".format(phase), loss, epoch
                )
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())

    model.eval()

    # get encoded layer

    def build_encoded_layer(loader: torch.utils.data.DataLoader) -> np.ndarray:
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


def print_losses(accuracy_list: List, phase: int) -> None:
    print("phase {}".format(phase))
    print(
        "validation loss mean: {} {}".format(
            np.mean(accuracy_list[0]),
            np.mean(accuracy_list[len(accuracy_list) - 1]),
        )
    )
    print(
        "validation loss variance: {} {}".format(
            np.var(accuracy_list[0]),
            np.var(accuracy_list[len(accuracy_list) - 1]),
        )
    )


def extract_model_input(
    df: pd.DataFrame,
    batch_size: int,
    device: torch.DeviceObjType,
    encoder_model: torch.nn.Module,
    encoded_features_count: int = 50,
) -> None:
    base_columns = [
        "date",
        "weight",
        "resp_1",
        "resp_2",
        "resp_3",
        "resp_4",
        "resp",
        "w_date",
    ]
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
        columns=base_columns + original_columns + encoded_columns,
    )
    new_df.to_csv(ENCODED_CSV)

    def save_part_to_csv(_new_df, new_data):
        _new_df = pd.DataFrame(
            new_data,
            columns=base_columns + original_columns + encoded_columns,
        )
        _new_df.to_csv(ENCODED_CSV, mode="a", header=False)

    for index, row in df.iterrows():
        if len(new_data) > 100000:
            save_part_to_csv(new_df, new_data)
            new_data = []
        a = row[
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN
        ].values
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
        nrows=1500,
    )
    if WEIGHTED_RESPS:
        new_df = new_df[new_df["weight"] != 0]
        new_df = new_df.reset_index(drop=True)

    pd.testing.assert_frame_equal(
        new_df.loc[0:899, "date":"feature_129"],
        df.loc[0:899, "date":"feature_129"],
        check_dtype=False,
        check_exact=False,
        check_less_precise=3,
    )
    test_new_df_source = new_df.loc[
        200 : 199 + batch_size,
        JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
    ]
    test_new_df_encode = new_df.loc[
        200 : 199 + batch_size,
        "enc_feature_0" : "enc_feature_{}".format(ENCODED_FEATURES_COUNT - 1),
    ]
    model_input = torch.from_numpy(np.array(test_new_df_source)).float().to(device)
    z = encoder_model.encoder(model_input)
    test_new_df_fresh_encode = pd.DataFrame(
        z.cpu().detach().numpy(),
        columns=encoded_columns,
    )
    test_new_df_fresh_encode = test_new_df_fresh_encode.astype(FLOAT_SIZE)
    test_new_df_encode = test_new_df_encode.astype(FLOAT_SIZE)
    pd.testing.assert_frame_equal(
        test_new_df_encode.loc[
            200 : 199 + batch_size,
            "enc_feature_0" : "enc_feature_{}".format(ENCODED_FEATURES_COUNT - 1),
        ].reset_index(drop=True),
        test_new_df_fresh_encode.loc[
            :, "enc_feature_0" : "enc_feature_{}".format(ENCODED_FEATURES_COUNT - 1)
        ].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        check_less_precise=3,
    )


def create_autencoder_and_train(
    batch_size: int = 200,
    effective_train_data: int = 1000000,
) -> torch.nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, n_epoch, validation_size = get_df_from_source(CSV_PATH, effective_train_data)
    df = add_time_columns_to_df(df)
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
        input_size=JaneStreetEncode1Dataset.Y_LEN,
        output_size=JaneStreetEncode1Dataset.Y_LEN,
        bottleneck=ENCODED_FEATURES_COUNT,
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

    print_losses(accuracy_list, 1)

    return model1


class JaneStreetDatasetPredict(Dataset):

    Y_LEN = 5

    # Constructor with defult values
    def __init__(self, df, device, transform=None, batch_size=None):
        """
        df.insert(3, "trade", None)
        df.loc[df["resp"] <= 0, "trade"] = 0
        df.loc[df["resp"] > 0, "trade"] = 1
        for one in range(4):
            i = 4 - one
            resp_str = "resp_{}".format(i)
            trade_str = "trade_{}".format(i)
            df.insert(3, trade_str, None)
            df.loc[df[resp_str] <= 0, trade_str] = 0
            df.loc[df[resp_str] > 0, trade_str] = 1
        """

        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device
        self.batch_size = batch_size

    # Getter
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = (
            torch.tensor(
                self.df.iloc[index][
                    "enc_feature_0" : "enc_feature_{}".format(
                        ENCODED_FEATURES_COUNT - 1
                    )
                ]
            )
            .float()
            .to(self.device)
        )
        y = torch.tensor(self.df.iloc[index]["resp_1":"resp"]).float().to(self.device)
        assert len(y) == JaneStreetDatasetPredict.Y_LEN

        item = (x, y)
        if self.transform:
            item = self.transform(item)
        return item

    # Get Length
    def __len__(self):
        return self.len


def get_core_model(
    input_size: int,
    output_size: int,
    hidden_count: int,
    dropout_p: float = 0.16,
    net_width: int = 32,
) -> torch.nn.Module:
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


class CustomSmoothL1Loss(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = torch.clone(target)
        # where we have missmatch in signs - meaning we missed critically - we will make the error bigger
        target[input * target < 0] = target[input * target < 0] * 3
        ret = super().forward(input, target)
        return ret


def train_model_predict(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    n_epoch: int = None,
    phase: int = None,
) -> Tuple[dict, dict]:
    writer = SummaryWriter()
    criterion = CustomSmoothL1Loss()
    # criterion = nn.SmoothL1Loss()
    if not LBFGS:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
                if not JANE_STREET_SUBMISSION:
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
                if not JANE_STREET_SUBMISSION:
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
                if z[one][-1] * y_test[one][-1] > 0:
                    accurate_guess += 1
                if z[one][-1] * rand_choice > 0:
                    accurate_guess_from_random += 1

                if torch.mean(z[one][:]) * y_test[one][-1] > 0:
                    accurate_guess_from_mean += 1

                if torch.median(z[one][:]) * y_test[one][-1] > 0:
                    accurate_guess_from_median += 1

            loss = criterion(z, y_test.float())
            if not JANE_STREET_SUBMISSION:
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


def batchify(row: np.ndarray, batch_size: int) -> np.ndarray:
    row = list(row)
    ret = []
    for one in range(batch_size):
        ret.append(row)
    ret = np.array(ret)
    return ret


def model_input_from_row(
    row: np.ndarray, batch_size: int, device: torch.DeviceObjType
) -> torch.nn.Module:
    batch_of_rows = batchify(row, batch_size)
    model_input = torch.from_numpy(batch_of_rows).float().to(device)
    return model_input


def visual_control_predict(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.DeviceObjType,
    df: pd.DataFrame,
):
    model.eval()
    index = np.random.randint(1, len(df))
    row = df.iloc[index][
        "enc_feature_0" : "enc_feature_{}".format(ENCODED_FEATURES_COUNT - 1)
    ].values
    model_input = model_input_from_row(row, batch_size, device)
    z = model(model_input)
    print("Model output")
    print(z[0])
    print("actual_data")
    print(df.iloc[index]["resp_1":"resp"])


def create_and_train_predict_model(
    df: pd.DataFrame,
    validation_size: int = 1000,
    batch_size: int = 200,
    n_epoch: int = 2,
) -> torch.nn.Module:
    # assert torch.cuda.is_available()
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
        ENCODED_FEATURES_COUNT,
        JaneStreetDatasetPredict.Y_LEN,
        hidden_count=3,
        dropout_p=0.2,
        net_width=1000,
    )

    model = model.float()
    model.to(device)

    train_res = train_model_predict(
        model,
        train_loader1,
        validation_loader1,
        n_epoch=n_epoch,
        phase=4,
    )
    (accuracy_list, loss_list) = train_res

    print_losses(accuracy_list, 1)

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
        df = add_time_columns_to_df(df)
        df = strip_for_batch_size(df, batch_size)
        extract_model_input(
            df,
            batch_size,
            DEVICE,
            encoder_model,
            encoded_features_count=ENCODED_FEATURES_COUNT,
        )
        df = None
    df, n_epoch, validation_size = get_df_from_source(ENCODED_CSV, effective_train_data)

    predict_model = create_and_train_predict_model(
        df,
        validation_size=validation_size,
        batch_size=batch_size,
        n_epoch=n_epoch,
    )
    print("Done")
    if JANE_STREET_SUBMISSION:
        return encoder_model, predict_model


if not JANE_STREET_SUBMISSION and __name__ == "__main__":
    fire.Fire(main)

if JANE_STREET_SUBMISSION:
    import janestreet

    BATCH_SIZE = 50
    encoder_model, predict_model = main(
        effective_train_data=1500, batch_size=BATCH_SIZE
    )
    encoder_model.eval()
    predict_model.eval()
    print("models ready")
    env = janestreet.make_env()
    for (test_df, pred_df) in tqdm(env.iter_test()):
        test_df = add_time_columns_to_df(test_df, eval=True)
        if not test_df["weight"].item() > 0:
            pred_df.action = 0
            env.predict(pred_df)
            continue
        row = test_df.loc[
            :,
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
        ]
        row = row.iloc[0]
        row = model_input_from_row(row.values, BATCH_SIZE, DEVICE)
        z = predict_model(encoder_model.encoder(row))

        pred_df.action = 1 if torch.mean(z[0]) > 0.0 else 0
        env.predict(pred_df)

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session