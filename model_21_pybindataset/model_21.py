"""
    Jan 30. 2021 - https://www.kaggle.com/c/jane-street-market-prediction Jane Street Market Prediction

    Goal of this model is to remove noise from given features and extract better, more
    stable features using autoencoder - and then use these features to predict
    trade decisions with simple MLP


    papers used:
    Autoencoders that don’t overfit towards the Identity
    https://proceedings.neurips.cc/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf
    and
    Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion
    https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Callable, List
from sklearn.preprocessing import MinMaxScaler

import os

JANE_STREET_SUBMISSION = False

if not JANE_STREET_SUBMISSION:
    import fire

if JANE_STREET_SUBMISSION:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    CSV_PATH = "/kaggle/input/jane-street-market-prediction/train.csv"
else:
    CSV_PATH = "/home/vaclavmatejka/devel/janestreet/jane-street-market-prediction/pybindataset.hdf"

ENCODED_CSV = "encoded.csv"
USE_FINISHED_ENCODE = (
    False  # if True run just prediction model and use the old ENCODED_CSV file
)
LBFGS = False  # if True Broyden–Fletcher–Goldfarb–Shanno algorithm else Adam
FLOAT_SIZE = "float64"
DATE_OVERFIT_FILL = None  # this setting is only good for overfitting the LB
# - final dates are going to be detached from training ones

try:
    if not USE_FINISHED_ENCODE:
        os.remove(ENCODED_CSV)
except FileNotFoundError:
    pass

# assert torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
ENCODED_FEATURES_COUNT = 400
FEATURE_STRING = "enc_feature_"

resp_start = "trade_in_3h_ETHBUSD_open_price"
resp_end = "trade_in_1h_BTCBUSD_open_price"

last_min_max_scaler = None


def get_file_size(filename: str) -> int:
    with open(filename) as f:
        return sum(1 for line in f)


def preprocessing_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    global last_min_max_scaler
    start_i = JaneStreetEncode1Dataset.Y_START_COLUMN
    end_i = JaneStreetEncode1Dataset.Y_END_COLUMN
    if not last_min_max_scaler:
        last_min_max_scaler = MinMaxScaler()
        df.loc[:, start_i:end_i] = last_min_max_scaler.fit_transform(
            df.loc[:, start_i:end_i]
        )
    else:
        df.loc[:, start_i:end_i] = last_min_max_scaler.transform(
            df.loc[:, start_i:end_i]
        )
    return df


def get_df_from_source() -> (pd.DataFrame, int, int):
    df = pd.read_hdf(CSV_PATH)
    df = df.astype(FLOAT_SIZE)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df.reset_index(drop=True)
    validation_size = int(df.shape[0] / 5)
    return df, validation_size


def get_autoencoded_df_from_source() -> (pd.DataFrame, int, int):
    df = pd.read_csv("encoded.csv")
    df = df.astype(FLOAT_SIZE)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df.reset_index(drop=True)
    validation_size = int(df.shape[0] / 5)
    return df, validation_size


def strip_for_batch_size(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    strip = df.shape[0] % batch_size
    return df.iloc[: df.shape[0] - strip]


class AddNoiseMixin:
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.15)

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
    Y_START_COLUMN = "BTCBUSD_open_price"
    Y_END_COLUMN = "ETHBUSD_taker_buy_quote_asset_volume_cov_128"

    Y_LEN = 1296

    @staticmethod
    def get_y_from_df(df: pd.DataFrame, index: int) -> Tensor:
        y = torch.from_numpy(
            df.iloc[index][
                JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN
            ].values
        )
        return y

    def __init__(
        self, df: pd.DataFrame, device: torch.DeviceObjType, transform: Callable = None
    ):
        self.init_dropout()
        self.df = df
        self.len = len(self.df)
        self.transform = transform
        self.device = device

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
            input_size, bottleneck, 1, dropout_p=0.5, net_width=1500
        )
        self.decoder = get_core_model(
            bottleneck, output_size, 1, dropout_p=0.5, net_width=1500
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = int(len(train_loader) / 4) * n_epoch
    adam_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1)
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        print("Train encoder model:")
        for x, y, mask in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y.float(), mask)
            if not JANE_STREET_SUBMISSION:
                writer.add_scalar("Train data phase: {}".format(phase), loss, epoch)
            loss.backward()
            optimizer.step()
            adam_scheduler.step()
            if not loss_list.get(epoch):
                loss_list[epoch] = []
            loss_list[epoch].append(loss.data.tolist())
        # perform a prediction on the validation data
        print("Validation encoder model:")
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

    base_columns = list(df.loc[:, resp_start:resp_end].columns)
    original_columns = list(
        df.loc[
            :,
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
        ].columns
    )
    encoded_columns = []
    for one in range(encoded_features_count):
        encoded_columns.append("{}{}".format(FEATURE_STRING, one))

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

    for index, row in tqdm(df.iterrows()):
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

    pd.testing.assert_frame_equal(
        new_df.loc[
            0:899,
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
        ],
        df.loc[
            0:899,
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
        ],
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
        "{}0".format(FEATURE_STRING) : "{}{}".format(
            FEATURE_STRING, ENCODED_FEATURES_COUNT - 1
        ),
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
            "{}0".format(FEATURE_STRING) : "{}{}".format(
                FEATURE_STRING, ENCODED_FEATURES_COUNT - 1
            ),
        ].reset_index(drop=True),
        test_new_df_fresh_encode.loc[
            :,
            "{}0".format(FEATURE_STRING) : "{}{}".format(
                FEATURE_STRING, ENCODED_FEATURES_COUNT - 1
            ),
        ].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        check_less_precise=3,
    )


def create_autencoder_and_train(
    batch_size: int = 200,
    effective_train_data: int = 1000000,
    n_epoch: int = 3,
) -> torch.nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, validation_size = get_df_from_source()

    df = preprocessing_scale_df(df)
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

    Y_LEN = 18

    # Constructor with defult values
    def __init__(self, df, device, transform=None, batch_size=None):
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
                    "{}0".format(FEATURE_STRING) : "{}{}".format(
                        FEATURE_STRING, ENCODED_FEATURES_COUNT - 1
                    )
                ]
            )
            .float()
            .to(self.device)
        )
        y = (
            torch.tensor(self.df.iloc[index][resp_start:resp_end])
            .float()
            .to(self.device)
        )
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
    """
    this custom loss should make the learning behave partially like regression and partially like classification
    """

    def forward(self, input: Tensor, target: Tensor, classification_weight=1) -> Tensor:
        target = torch.clone(target)

        distance = torch.abs(input - target)
        zero_pass_factor = torch.ones_like(distance)
        zero_pass_factor[input * target < 0] = (
            zero_pass_factor[input * target < 0] * classification_weight
        )
        # zero_pass_factor[:, -1] = zero_pass_factor[:, -1] * 10

        distance[distance < 1] = torch.exp(distance[distance < 1])
        distance[distance > 1] = distance[distance > 1] + np.e - 1
        distance = distance * zero_pass_factor
        distance = distance.float()

        target[input > target] = input[input > target] - distance[input > target]
        target[input < target] = input[input < target] + distance[input < target]

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
    if not LBFGS:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        step = int(len(train_loader) / 4) * n_epoch
        adam_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1)
    else:
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=15, history_size=60, lr=0.01
        )
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        classification_weight = np.sqrt(
            10 ** (n_epoch - epoch)
        )  # if set to 1, CustomSmoothL1Loss will act as a pure regression
        print("Train predict model:")
        for x, y in tqdm(train_loader):
            model.train()
            if not LBFGS:
                optimizer.zero_grad()
                z = model(x.float())
                loss = criterion(
                    z, y.float(), classification_weight=classification_weight
                )
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
                adam_scheduler.step()

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
        accurate_guess_mean = 0
        accurate_guess_mean_last3 = 0
        accurate_guess_from_random = 0
        accurate_guess_from_batch_noise_mean = 0
        accurate_guess_from_batch_noise_majority = 0
        accurate_guess_from_batch_noise_mean_majority = 0

        total_count = 1

        print("Validation predict model:")
        for x_test, y_test in tqdm(validation_loader):
            model.eval()
            z = model(x_test.float())

            for z_index in range(len(z)):
                total_count += 1
                rand_choice = int(torch.randint(0, 2, (1, 1))[0])
                if rand_choice == 0:
                    rand_choice = -1
                if z[z_index][-1] * y_test[z_index][-1] > 0:
                    accurate_guess += 1
                if torch.mean(z[z_index]) * y_test[z_index][-1] > 0:
                    accurate_guess_mean += 1
                if torch.mean(z[z_index][2:]) * y_test[z_index][-1] > 0:
                    accurate_guess_mean_last3 += 1
                if z[z_index][-1] * rand_choice > 0:
                    accurate_guess_from_random += 1

                """
                bagging attempts:

                noisy_x_test = add_noise_to_model_input_tensor(
                    batchify_to_tensor(x_test[z_index], 11)
                )
                noisy_z = model(noisy_x_test.float())
                trade = get_trade_from_noisy_mean(noisy_z)
                if trade * y_test[z_index][-1] > 0:
                    accurate_guess_from_batch_noise_mean += 1

                trade = get_trade_from_noisy_majority(noisy_z)
                if trade * y_test[z_index][-1] > 0:
                    accurate_guess_from_batch_noise_majority += 1

                trade = get_trade_from_noisy_mean_majority(noisy_z)
                if trade * y_test[z_index][-1] > 0:
                    accurate_guess_from_batch_noise_mean_majority += 1
                """

            loss = criterion(
                z, y_test.float(), classification_weight=classification_weight
            )
            if not JANE_STREET_SUBMISSION:
                writer.add_scalar(
                    "Validation data phase: {}".format(phase), loss, epoch
                )
            if not accuracy_list.get(epoch):
                accuracy_list[epoch] = []
            accuracy_list[epoch].append(loss.data.tolist())
        print("epoch_{} accuracy: {}".format(epoch, accurate_guess / total_count))
        print(
            "epoch_{} accuracy mean: {}".format(
                epoch, accurate_guess_mean / total_count
            )
        )
        print(
            "epoch_{} accuracy mean last3: {}".format(
                epoch, accurate_guess_mean_last3 / total_count
            )
        )
        print(
            "epoch_{} accuracy from random choice: {}".format(
                epoch, accurate_guess_from_random / total_count
            )
        )
        print(
            "epoch_{} accuracy from noisy mean: {}".format(
                epoch, accurate_guess_from_batch_noise_mean / total_count
            )
        )
        print(
            "epoch_{} accuracy from noisy majority: {}".format(
                epoch, accurate_guess_from_batch_noise_majority / total_count
            )
        )
        print(
            "epoch_{} accuracy from noisy mean majority: {}".format(
                epoch, accurate_guess_from_batch_noise_mean_majority / total_count
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


def batchify_to_tensor(row: Tensor, batch_size: int) -> Tensor:
    row = row.cpu().detach().numpy()
    batch_of_rows = batchify(row, batch_size)
    return torch.from_numpy(batch_of_rows).float().to(DEVICE)


def model_input_from_row(
    row: np.ndarray,
    batch_size: int,
) -> Tensor:
    batch_of_rows = batchify(row, batch_size)
    model_input = torch.from_numpy(batch_of_rows).float().to(DEVICE)
    return model_input


def add_noise_to_model_input_tensor(batch_of_rows: Tensor) -> Tensor:
    ret = []
    for row in batch_of_rows:
        noise = torch.randn_like(row)
        noise = noise.multiply(1 / 4)
        row = row + noise
        row = row.cpu().detach().numpy()
        ret.append(row)
    return torch.from_numpy(np.array(ret)).float().to(DEVICE)


def get_trade_from_noisy_majority(z: Tensor) -> int:
    trades = []
    for row in z:
        trade = 1 if row[-1] > 0 else -1
        trades.append(trade)
    return 1 if sum(trades) > 0 else -1


def get_trade_from_noisy_mean_majority(z: Tensor) -> int:
    trades = []
    for row in z:
        row_mean = torch.mean(row[:])
        trade = 1 if row_mean > 0 else -1
        trades.append(trade)
    return 1 if sum(trades) > 0 else -1


def get_trade_from_noisy_mean(z: Tensor) -> int:
    z = z.mean(0)
    res = torch.mean(z[:])  # first is weight, skip it and go from resp_1 to resp
    return 1 if res > 0 else -1


def visual_control_predict(
    model: torch.nn.Module,
    batch_size: int,
    df: pd.DataFrame,
):
    model.eval()
    index = np.random.randint(1, len(df))
    row = df.iloc[index][
        "{}0".format(FEATURE_STRING) : "{}{}".format(
            FEATURE_STRING, ENCODED_FEATURES_COUNT - 1
        )
    ].values
    model_input = model_input_from_row(row, batch_size)
    z = model(model_input)
    print("Model output")
    print(z[0])
    print("actual_data")
    print(df.iloc[index][resp_start:resp_end])


def create_and_train_predict_model(
    df: pd.DataFrame,
    validation_size: int = 1000,
    batch_size: int = 200,
    n_epoch: int = 2,
) -> torch.nn.Module:
    # assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = df.fillna(df.mean())
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
        dropout_p=0.5,
        net_width=1500,
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
            n_epoch=8,
        )

        df, validation_size = get_df_from_source()
        df = preprocessing_scale_df(df)
        df = strip_for_batch_size(df, batch_size)
        extract_model_input(
            df,
            batch_size,
            DEVICE,
            encoder_model,
            encoded_features_count=ENCODED_FEATURES_COUNT,
        )
        df = None
    df, validation_size = get_autoencoded_df_from_source()

    predict_model = create_and_train_predict_model(
        df,
        validation_size=validation_size,
        batch_size=batch_size,
        n_epoch=12,
    )
    print("Done")
    if JANE_STREET_SUBMISSION:
        return encoder_model, predict_model


if not JANE_STREET_SUBMISSION and __name__ == "__main__":
    fire.Fire(main)

if JANE_STREET_SUBMISSION:
    import janestreet

    BATCH_SIZE = 500
    encoder_model, predict_model = main(
        effective_train_data=400000, batch_size=BATCH_SIZE
    )
    encoder_model.eval()
    predict_model.eval()
    print("models ready")
    env = janestreet.make_env()
    print("janestreet env created")
    for (test_df, pred_df) in tqdm(env.iter_test()):
        test_df = preprocessing_scale_df(test_df)
        if not test_df["weight"].item() > 0:
            pred_df.action = 0
            env.predict(pred_df)
            continue

        row = test_df.loc[
            :,
            JaneStreetEncode1Dataset.Y_START_COLUMN : JaneStreetEncode1Dataset.Y_END_COLUMN,
        ]
        row = row.iloc[0]
        batch_of_rows = model_input_from_row(row.values, 1)
        z = predict_model(encoder_model.encoder(batch_of_rows))
        term = torch.mean(z[0])
        pred_df.action = 1 if term >= 0 else 0

        env.predict(pred_df)
