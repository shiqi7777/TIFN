from pytorch_lightning.utilities.cli import LightningCLI

from TIFN.datamodule import CROHMEDatamodule
from TIFN.lit_tifn import LitTIFN
from TIFN.callback.log_prediction import LogPredictionSamples
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    LightningCLI(
        LitTIFN,
        CROHMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={
            "logger": TensorBoardLogger("lightning_logs", name=""),
            "callbacks": [LogPredictionSamples(max_samples=16, max_batches=4)],
        },
    )
if __name__ == "__main__":
    main()
