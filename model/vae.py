import torch
import typing

class VAE(torch.nn.Module):
	def __init__(self, input_dim: int):
		super().__init__()
		self.input_dim = input_dim

		hidden_dim = 400
		latent_dim = 20
	
		# Encoder
		self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim)
		self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
		self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)

		# Decoder
		self.fc3 = torch.nn.Linear(latent_dim, hidden_dim)
		self.fc4 = torch.nn.Linear(hidden_dim, self.input_dim)

	def encode(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		h1 = torch.nn.functional.relu(self.fc1(x))
		return self.fc_mu(h1), self.fc_logvar(h1)
	
	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	
	def decode(self, x: torch.Tensor):
		h3 = torch.nn.functional.relu(self.fc3(x))
		return self.fc4(h3)
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x: torch.Tensor):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon_x = self.decode(z)
		return recon_x, mu, logvar

class VAELoss(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, y: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
		BCE = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, y, reduction='sum')
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD