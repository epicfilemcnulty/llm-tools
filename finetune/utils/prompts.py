def chatml_prompt(row):
    convos = row["conversations"]
    text = ""
    for convo in convos:
        if convo["from"] == "system":
            text = text + "<|im_start|>system\n" + convo["value"] + "<|im_end|>\n"
        if convo["from"] == "human":
            text = text + "<|im_start|>user\n" + convo["value"] + "<|im_end|>\n"
        if convo["from"] == "gpt":
            text = text + "<|im_start|>assistant\n" + convo["value"] + "<|im_end|>\n"
    return text

def vicuna_prompt(row):
    convos = row["conversations"]
    text = ""
    for convo in convos:
        if convo["from"] == "system":
            text = text + "SYSTEM: " + convo["value"] + "\n"
        if convo["from"] == "human":
            text = text + "USER: " + convo["value"] + "\n"
        if convo["from"] == "gpt":
            text = text + "ASSISTANT: " + convo["value"] + "\n</s>"
    return text
