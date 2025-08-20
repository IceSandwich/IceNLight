from datasets import load_dataset
import json
import argparse
import typing

class Args:
	hf_repo: str
	output: str
	dataset_format: str
	replace_instruction: str | None = None

	FORMAT_ALPACA = "alpaca"

def SetupArgs() -> Args:
	parser = argparse.ArgumentParser()
	parser.add_argument("--hf_repo", type=str, required=True)
	parser.add_argument("--output", type=str, required=True, help="Output filename like `data/dataset/llm/custom_dataset.json`")
	parser.add_argument("--dataset_format", type=str, choices=["alpaca"], required=True)
	parser.add_argument("--replace_instruction", type=str, required=False, default=None)
	return parser.parse_args()

def postprocess(dataItem: typing.Dict[str, str], format: str) -> typing.Optional[typing.Dict[str, str]]:
	"""
	Write your own postprocessing function here. Return None to skip this item.
	"""
	return dataItem

def main(args: Args):
	# 1. 加载 Hugging Face 上的数据集
	dataset = load_dataset(args.hf_repo, split="train")
	print("Loading dataset:", dataset)

	# 2. 格式化数据集
	flatten: typing.List[typing.Dict[str, str]] = []
	if args.dataset_format == Args.FORMAT_ALPACA:
		for item in dataset.to_list():
			item: typing.Dict[str, str]
			if args.replace_instruction is not None:
				item["instructions"] = args.replace_instruction

			if "instruction" in item:
				if len(item["instruction"].strip()) == 0 and len(item["input"].strip())== 0:
					continue
			else:
				if len(item["input"].strip()) == 0:
					continue

			new_item = postprocess(item, args.dataset_format)
			if new_item is not None:
				flatten.append(new_item)
	else:
		raise ValueError(f"`{args.dataset_format}` is not supported.")
	
	# 3. 保存数据集
	if args.output.endswith(".jsonl"):
		with open(args.output, "w", encoding="utf-8") as f:
			for item in flatten:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
	elif args.output.endswith(".json"):
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(json.dumps(flatten, indent=4))
	
	print(f"- Dataset items: {len(flatten)}")
	print(f"- Save to {args.output}")

if __name__ == "__main__":
	args = SetupArgs()
	main(args)