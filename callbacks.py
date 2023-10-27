from pytorch_lightning.callbacks import EarlyStopping, Callback
import pytorch_lightning as pl

class MyPrintingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        print('Starting to train')

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('Tringing is done')
