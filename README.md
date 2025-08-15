# Usage

1. Generate training configuration

``` bash
python main.py fit --trainer.accelerator="auto" --trainer.devices="auto" --trainer.strategy="auto" --trainer.accumulate_grad_batches=1 --trainer.max_epochs=100 --trainer.precision="bf16" --print_config > train.yaml
```

Adjust the configuration as you like in `train.yaml`.

2. Train the model

``` bash
python main.py fit --config=train.yaml
```