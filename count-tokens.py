import sys
from transformers import AutoTokenizer


args = sys.argv[1:]

tokenizer = AutoTokenizer.from_pretrained(args[0])
file = open(args[1], "r")
content = file.read()
file.close()

input_ids = tokenizer(content, return_tensors="pt").input_ids[0]
print(len(input_ids))
