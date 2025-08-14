from pytorch_lightning.cli import LightningCLI
import icenlight

from model.main import ModelModule
from data.main import DataModule

def cli_main():
	cli = LightningCLI(ModelModule, DataModule)

if __name__ == "__main__":
	cli_main()