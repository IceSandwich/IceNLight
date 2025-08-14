import pytorch_lightning as pl
import icenlight
import typing
import torch
import dataclasses

class LossModule(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self):
		pass

@dataclasses.dataclass
class TrainConfig:
	seed: typing.Union[int, None] = None
	learning_rate: float = 1e-3

@dataclasses.dataclass
class ModelConfig:
	pass

class ModelResult:
	pass

class ModelModule(pl.LightningModule):
	def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
		self.train_config = train_config
		self.model_config = model_config
		self.init_module()
		self.save_hyperparameters()

		self.build_components()

	def init_module(self):
		if self.train_config.seed is not None:
			pl.seed_everything(self.train_config.seed)

	def training_step(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor], batch_idx: int):
		result = self.forward_batch(batch)
		loss = self.calculate_loss(result)
		if type(loss) == torch.Tensor:
			self.log(icenlight.Graph_TrainingLoss, loss.item(), prog_bar=True)
			return loss
		else:
			self.log_dict(loss)
			return loss["loss"]
	
	def on_train_batch_start(self):
		self.log(icenlight.Graph_Epoch, self.current_epoch, on_step=True)
	
	def validation_step(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor], batch_idx: int) -> None:
		result = self.forward_batch(batch)
		loss = self.calculate_loss(result)
		self.log(icenlight.Graph_ValidationLoss, loss.item(), prog_bar=True)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate)
		return optimizer
	
	
	def build_components(self):
		pass

	def forward_batch(self, batch: typing.Union[typing.Dict[str, typing.Any], torch.Tensor]) -> ModelResult:
		pass

	def calculate_loss(self, result: ModelResult) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
		pass