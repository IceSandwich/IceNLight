import pytorch_lightning as pl
import icenlight
import typing
import torch
import dataclasses
from .vae import VAE, VAELoss

@dataclasses.dataclass
class TrainConfig:
	learning_rate: float = 1e-3

@dataclasses.dataclass
class ModelConfig:
	pass

class ModelResult:
	recon_x: torch.Tensor
	mu: torch.Tensor
	logvar: torch.Tensor

class ModelModule(pl.LightningModule):
	def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
		super().__init__()

		self.train_config = train_config
		self.model_config = model_config
		self.save_hyperparameters()

		self.build_components()

	def training_step(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor], batch_idx: int):
		result = self.forward_batch(batch)
		loss = self.calculate_loss(batch, result)
		if type(loss) == torch.Tensor:
			self.log(icenlight.Graph_TrainingLoss, loss.item(), prog_bar=True)
			return loss
		else:
			self.log_dict(loss)
			return loss["loss"]
	
	def on_train_batch_start(self, batch: typing.Any, batch_idx: int):
		self.log(icenlight.Graph_Epoch, float(self.current_epoch), on_step=True)
	
	def validation_step(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor], batch_idx: int) -> None:
		result = self.forward_batch(batch)
		loss = self.calculate_loss(batch, result)
		self.log(icenlight.Graph_ValidationLoss, loss.item(), prog_bar=True)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate)
		return optimizer
	
	def build_components(self):
		self.model = VAE(28 * 28)
		self.loss = VAELoss()

	def preprocess_batch(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor]) -> torch.Tensor:
		x = batch[0].squeeze(dim=1)
		x = x.view(-1, 28*28)
		return x

	def forward_batch(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor]) -> ModelResult:
		x = self.preprocess_batch(batch)
		ret = ModelResult()
		ret.recon_x, ret.mu, ret.logvar = self.model(x)
		return ret

	def calculate_loss(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor], result: ModelResult) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
		x = self.preprocess_batch(batch)
		loss = self.loss(x, result.recon_x, result.mu, result.logvar)
		return loss