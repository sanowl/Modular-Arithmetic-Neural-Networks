from datasets import load_dataset

# Load the dataset
raw_dataset = load_dataset('math_dataset', 'arithmetic__add_or_sub', split='train', trust_remote_code=True)

# Print a few samples to understand the structure
for i in range(10):
    print(raw_dataset[i])
