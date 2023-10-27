from dataset import MnistDataModule
from model import NN
import torch
import pytorch_lightning as pl
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    logger = TensorBoardLogger('tb_logs', name='mnist_models_v1')
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('tb_logs/profiler0'),
        schedule = torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = NN(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    ).to(config.device)
    trainer = pl.Trainer(
        profiler=profiler,
        min_epochs=1, 
        max_epochs=config.NUM_EPOCHS, 
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor='val_loss')],
        logger=logger
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
