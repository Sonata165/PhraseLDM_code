from pytorch_lightning.cli import LightningCLI
import random
import numpy as np
import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision("highest") # highest | high | medium

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--test", default=False, action="store_true", help="Run test instead of training.")
        parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint path for testing.")

# if __name__ == "__main__":
#     # Call seed_everything at the top-level
#     seed_everything(42)

#     cli = MyLightningCLI(run=False)

#     if cli.config.test:
#         cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
#     else:
#         cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    seed_everything(42)
    cli = MyLightningCLI(run=False)

    if cli.config.test:
        cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    else:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
        # 自动测试最优模型
        best_ckpt_path = None
        if hasattr(cli.trainer, "checkpoint_callback") and cli.trainer.checkpoint_callback is not None:
            best_ckpt_path = cli.trainer.checkpoint_callback.best_model_path
        elif hasattr(cli.trainer, "callbacks"):
            for cb in cli.trainer.callbacks:
                if hasattr(cb, "best_model_path"):
                    best_ckpt_path = cb.best_model_path
                    break
        if best_ckpt_path and best_ckpt_path != "":
            print(f"Testing with best checkpoint: {best_ckpt_path}")
            cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=best_ckpt_path)
        else:
            print("No best checkpoint found, running test with current model weights.")
            cli.trainer.test(model=cli.model, datamodule=cli.datamodule)