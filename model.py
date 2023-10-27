import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        # accuracy = self.accuracy(scores, y)
        # f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                # "train_accuracy": accuracy,
                # "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
        )
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image('mnist_image', grid, self.global_step)
        return {"loss": loss, "scores": scores, "y": y}
    
    def on_train_epoch_end(self, outputs):
        scores = torch.cat([x['scores'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        self.log_dict(
            {
                'train_acc': self.accuracy(scores, y),
                'train_f1': self.f1_score(scores, y)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
        )


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)