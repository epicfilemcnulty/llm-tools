from exllamav2.generator import ExLlamaV2Sampler

def exl2_query(query, sampler, tokenizer, generator, lora, sc=[]):
 
    stop_conditions = [tokenizer.eos_token_id] + sc
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = sampler["temperature"]
    settings.top_k = sampler["top_k"]
    settings.top_p = sampler["top_p"]
    settings.min_p = sampler["min_p"]
    settings.token_repetition_penalty = sampler["repetition_penalty"]

    input_ids = tokenizer.encode(query, add_bos = sampler["add_bos"])
    prompt_tokens = input_ids.shape[-1]
 
    generator.set_stop_conditions(stop_conditions)
    generator.begin_stream(input_ids, settings, loras = lora)
    generated_tokens = 0
    new_text = ""
    while True:
        chunk, eos, tokens = generator.stream()
        generated_tokens += 1
        new_text += chunk
        if eos or generated_tokens == max_new_tokens:
            break

    stop_reason = "eos" if eos else "length"
    return new_text, prompt_tokens, generated_tokens, stop_reason
