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
ENCODED_CSV = "encoded.csv"


def get_file_size(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def get_df_from_source(path, effective_train_data) -> (pd.DataFrame, int, int):
    min_epochs = 3
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
        return inputs, mask


class AddGausseNoiseMixin(AddNoiseMixin):
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.5)

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs)
        noise = noise.multiply(1 / 4)
        mask = self.get_mask(inputs)
        # mask - zeros means corrupted
        reversed_mask = self.reverse_mask(mask)
        # reversed_mask - zeros means noncorrupted
        noise = noise * reversed_mask
        # noise is zero when mask is uncorrupted/zero
        inputs = inputs + noise
        return inputs, mask


class JaneStreetEncode1Dataset(AddDropoutNoiseMixin, Dataset):
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.40)

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


class BaseJaneStreetEncode2Dataset(Dataset):

    # Constructor with defult values
    def __init__(
        self,
        encoded_layer,
        device,
        transform=None,
    ):
        self.init_dropout()
        self.encoded_layer = encoded_layer
        self.len = len(encoded_layer)
        self.transform = transform
        self.device = device

    # Getter
    def __getitem__(self, index):
        inputs, mask = self.add_noise(torch.from_numpy(self.encoded_layer[index]))
        sample = (inputs, torch.from_numpy(self.encoded_layer[index]), mask)
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


class JaneStreetEncode2Dataset(AddDropoutNoiseMixin, BaseJaneStreetEncode2Dataset):
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.35)


class JaneStreetEncode3Dataset(AddDropoutNoiseMixin, BaseJaneStreetEncode2Dataset):
    def init_dropout(self):
        self.dropout_model = nn.Dropout(p=0.45)


def get_core_model(input_size, output_size, hidden_count, dropout_p=0.15, net_width=32):
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


class autoencoder(nn.Module):
    def __init__(
        self, small_number, big_number, dropout_p, input_size, output_size, bottleneck
    ):
        super(autoencoder, self).__init__()
        self.encoder = get_core_model(
            input_size, bottleneck, 1, dropout_p=dropout_p, net_width=big_number
        )
        self.decoder = get_core_model(
            bottleneck, output_size, 1, dropout_p=dropout_p, net_width=big_number
        )
        """
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
        """

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
    lr=None,
    phase=None,
    device=None,
):
    writer = SummaryWriter()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    criterion = EmphasizedSmoothL1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr)
    accuracy_list = {}
    loss_list = {}
    for epoch in range(n_epoch):
        for x, y, mask in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y.float(), mask)
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
            writer.add_scalar("Validation data phase: {}".format(phase), loss, epoch)
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


def visual_control(model, batch_size, device, df, validation_encoded_layer):
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


def extract_model_input(df, batch_size, device, e1, e2, e3, encoded_features_count=50):
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
            z = e1.encoder(model_input)
            z = e2.encoder(z)
            z = e3.encoder(z)
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
    z = e1.encoder(model_input)
    z = e2.encoder(z)
    z = e3.encoder(z)
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


def create_autencoder(
    big_number=1500,
    small_number=1500,
    dropout_p=0.15,
    batch_size=200,
    lr=0.20,
    effective_train_data=400000,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, n_epoch, validation_size = get_df_from_source(CSV_PATH, effective_train_data)
    print("n_epoch {}".format(n_epoch))
    frac = 1 / 1
    train_sample = df.iloc[:-validation_size]
    train_sample = train_sample.sample(frac=frac).reset_index()
    train_sample = strip_for_batch_size(train_sample, batch_size)
    validation_sample = df.iloc[-validation_size:]
    validation_sample = validation_sample.sample(frac=frac).reset_index()
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
        small_number=small_number,
        big_number=big_number,
        dropout_p=dropout_p,
        input_size=130,
        output_size=130,
        bottleneck=60,
    )
    model1 = model1.float()
    model1.to(device)

    train_res = train_model_encoder(
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
    )
    janestreet2_validation = JaneStreetEncode2Dataset(
        encoded_layer=validation_encoded_layer1,
        device=device,
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
    train_res = train_model_encoder(
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

    janestreet3_train = JaneStreetEncode3Dataset(
        encoded_layer=train_encoded_layer2,
        device=device,
    )
    janestreet3_validation = JaneStreetEncode3Dataset(
        encoded_layer=validation_encoded_layer2,
        device=device,
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

    train_res = train_model_encoder(
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

    visual_control(
        model3,
        batch_size,
        device,
        validation_sample,
        validation_encoded_layer2,
    )
    return model1, model2, model3, device


class JaneStreetDatasetPredict(Dataset):

    # Constructor with defult values
    def __init__(self, df, device, transform=None, batch_size=None):
        df.insert(3, "trade", None)
        df.loc[df["resp"] <= 0, "trade"] = -0.1
        df.loc[df["resp"] > 0, "trade"] = 0.1

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
            torch.tensor(self.df.iloc[index]["enc_feature_0":"enc_feature_49"])
            .float()
            .to(self.device)
        )
        y = torch.tensor(self.df.iloc[index]["trade":"resp"]).float().to(self.device)

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
    scale_factor=1,
    min_scale=64,
):
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
        old_net_width = net_width
        possible_net_width = int(net_width * scale_factor)
        if possible_net_width > min_scale:
            net_width = possible_net_width
        append_layer(layers, old_net_width, net_width)

    append_layer(layers, net_width, output_size, just_linear=True)
    return torch.nn.Sequential(*layers)


class CustomSmoothL1Loss(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = torch.clone(target)
        # where we have missmatch in signs - meaning we missed big - we will make the error way bigger
        target[input * target < 0] = target[input * target < 0] * 50
        # target[target == 10] = 30
        # target[target == -10] = -30
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
    criterion = CustomSmoothL1Loss()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = torch.optim.Adadelta(model.parameters())
    # optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.Adagrad(model.parameters())
    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr)
    # optimizer = torch.optim.Adam(model.parameters())
    accuracy_list = {}
    loss_list = {}
    scheduler_count = 0
    for epoch in range(n_epoch):
        for x, y in train_loader:
            scheduler_count += 1
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

            if scheduler_count % 40000 == 0:
                scheduler.step(scheduler_count / 40000)

        # perform a prediction on the validation data
        accurate_guess = 0
        total_count = 0

        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())

            for one in range(len(z)):
                # if y_test[one][1] > 0:
                total_count += 1
                if z[one][0] * y_test[one][0] > 0:
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


def visual_control_predict(model, batch_size, device, df):
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
    hidden_count = 3
    big_number = 1500
    model = get_core_model(
        50,
        7,
        hidden_count,
        dropout_p=0.15,
        net_width=big_number,
        scale_factor=1 / 1,
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
        phase=4,
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

    visual_control_predict(
        model,
        batch_size,
        device,
        validation_sample,
    )
    return model, device


def main(
    big_number=1500,
    small_number=1500,
    dropout_p=0.15,
    batch_size=200,
    lr=0.20,
    effective_train_data=500000,
):
    model_enc1, model_enc2, model_enc3, device = create_autencoder(
        big_number=big_number,
        small_number=small_number,
        dropout_p=dropout_p,
        batch_size=batch_size,
        lr=lr,
        effective_train_data=effective_train_data,
    )

    df, n_epoch, validation_size = get_df_from_source(CSV_PATH, effective_train_data)
    initial_df_size = df.shape[0]
    df = strip_for_batch_size(df, batch_size)
    extract_model_input(
        df,
        batch_size,
        device,
        model_enc1,
        model_enc2,
        model_enc3,
        encoded_features_count=50,
    )
    df, n_epoch, validation_size = get_df_from_source(ENCODED_CSV, effective_train_data)
    print("encoded df: {} initial df: {}".format(df.shape[0], initial_df_size))

    model = create_and_train_predict_model(
        df,
        validation_size=validation_size,
        batch_size=200,
        n_epoch=n_epoch,
        lr=0.05,
    )
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
