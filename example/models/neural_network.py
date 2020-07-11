from ml_api.contract import enums
from ml_api.server import InferenceModel
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO


# TODO: I'm not a DL person, so this just serves as an example
class Net(LightningModule):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(dim, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.pipe(x)

    def training_step(self, batch, *args):
        x, y = batch
        loss = self.loss(self.forward(x), y)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class NeuralNetworkModel(InferenceModel):
    def make_model(self, **kwargs):
        return Net(**kwargs)

    def serialize(self, model, x, y=None):
        b = BytesIO()
        torch.onnx.export(model, torch.empty(*x.shape), b)

        return b.getvalue()

    def fit(self, model: LightningModule, x, y=None, key=None, **kwargs):
        trainer = Trainer()

        ds = TensorDataset(torch.from_numpy(x.values).float(), torch.from_numpy(y.values).float())
        dl = DataLoader(ds, batch_size=kwargs.pop('batch_size', 1000))

        trainer.fit(model, train_dataloader=dl)

        return trainer.model

    def serializer_backend(self):
        return enums.SerializerBackend.ONNX