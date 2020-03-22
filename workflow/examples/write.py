from workflow import reporting

import numpy as np
import sacred
import torch
import torch.nn.functional as f
import tqdm


experiment = sacred.Experiment("Example experiment for the workflow package")


class RegressionNet(torch.nn.Module):
    def __init__(self, hdims: int):
        super().__init__()
        self._h1 = torch.nn.Linear(2, hdims)
        self._h2 = torch.nn.Linear(hdims, hdims)
        self._h3 = torch.nn.Linear(hdims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.leaky_relu(self._h1(x))
        x = f.leaky_relu(self._h2(x))
        x = self._h3(x)
        return x


class Trainer:
    def __init__(self, net: torch.nn.Module, learning_rate: float, batch_size: int):
        self._train_x = np.random.random(size=(500, 2)).astype(np.float32)
        self._train_y = (self._train_x[:, 0] * 3 + self._train_x[:, 1] +
                         np.random.randn(self._train_x.shape[0]).astype(np.float32) * 0.1)
        self._valid_x = np.random.random(size=(100, 2)).astype(np.float32)
        self._valid_y = (self._train_x[:, 0] * 3 + self._train_x[:, 1] +
                         np.random.randn(self._train_x.shape[0]).astype(np.float32) * 0.1)

        self._net = net
        self._optim = torch.optim.Adam(self._net.parameters(), learning_rate)
        self._batch_size = batch_size

        reporting.register_field("train_loss")
        reporting.register_field("valid_loss")

    def update(self):
        batch_indices = np.random.choice(self._train_x.shape[0], size=self._batch_size)
        batch_x = torch.from_numpy(self._train_x[batch_indices])
        batch_y_target = torch.from_numpy(self._train_y[batch_indices])

        loss = ((self._net(batch_x) - batch_y_target)**2).mean()
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        batch_indices = np.random.choice(self._valid_x.shape[0], size=self._batch_size)
        batch_x = torch.from_numpy(self._valid_x[batch_indices])
        batch_y_target = torch.from_numpy(self._valid_y[batch_indices])

        valid_loss = ((self._net(batch_x) - batch_y_target)**2).mean()
        reporting.iter_record("train_loss", loss.item())
        reporting.iter_record("valid_loss", valid_loss.item())


# noinspection PyUnusedLocal
@experiment.config
def _config():
    learning_rate = 1e-5
    batch_size = 128
    hdims = 50
    num_iterations = 100000
    reporting_dir = "/tmp/regression_reporting"


@experiment.capture
def build_trainer(hdims: int, learning_rate: float, batch_size: int):
    return Trainer(RegressionNet(hdims), learning_rate, batch_size)


@experiment.automain
def _run(reporting_dir: str, num_iterations: int):
    reporting.register_global_reporter(experiment, reporting_dir)

    trainer = build_trainer()
    reporting.finalize_fields()
    trange = tqdm.trange(num_iterations)
    for iteration in trange:
        trainer.update()

        reporting.iterate()
        trange.set_description(f"{iteration} -- " + reporting.get_description(["train_loss", "valid_loss"]))
