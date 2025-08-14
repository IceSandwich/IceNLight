import icenlight

import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST
import dataclasses

@dataclasses.dataclass
class DatasetConfig:
	batch_size: int = 8
	drop_last: bool = False
	num_worker: int = 4

class DataModule(pl.LightningDataModule):
	def __init__(self, train_config: DatasetConfig, validation_config: DatasetConfig, test_config: DatasetConfig):
		self.assets_dir = icenlight.GetAssetsDir("mnist")
		self.train_config = train_config
		self.validation_config = validation_config
		self.test_config = test_config

		self.save_hyperparameters()

	def prepare_data(self):
		MNIST(self.assets_dir, train=True, download=True)
		MNIST(self.assets_dir, train=False, download=True)

	def setup(self, stage: str) -> None:
		if stage == "fit":
			train_ds = MNIST(self.assets_dir, train=True)
			self.train_ds, self.val_ds = icenlight.RandomSplit(train_ds, [0.9, 0.1], seed=42)

		if stage == "test":
			self.test_ds = MNIST(self.assets_dir, train=False)

		if stage == "predict":
			pass
		
	def train_dataloader(self):
		return torch.utils.data.DataLoader(self.train_ds, batch_size=self.train_config.batch_size, shuffle=True, drop_last=self.train_config.drop_last, num_workers=self.train_config.num_worker)
	
	def val_dataloader(self):
		return torch.utils.data.DataLoader(self.val_ds, batch_size=self.validation_config.batch_size, shuffle=True, drop_last=self.validation_config.drop_last, num_workers=self.validation_config.num_worker)
	
	def test_dataloader(self):
		return torch.utils.data.DataLoader(self.test_ds, batch_size=self.test_config.batch_size, shuffle=True, drop_last=self.test_config.drop_last, num_workers=self.test_config.num_worker)
	
	def teardown(self, stage: str):
		if stage == "fit":
			pass

		if stage == "test":
			pass

		if stage == "predict":
			pass