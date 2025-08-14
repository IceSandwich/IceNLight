import torch
import typing

def RandomSplit(dataset: typing.Union[torch.utils.data.Dataset, typing.List[str]], ratio: typing.Union[typing.List[int], typing.List[float]], seed: typing.Optional[int] = None):
	generator = torch.Generator().manual_seed(seed=seed) if seed else None
	return torch.utils.data.random_split(dataset, ratio, generator)