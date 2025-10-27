from datasets import load_dataset

dataset = load_dataset("TeeZee/dolly-15k-pirate-speech", split="train")
df = dataset.to_pandas()
df = df[["instruction", "response"]]
output_file = "dolly_pirate_simple_dataset.csv"
df.to_csv(output_file, index=False)