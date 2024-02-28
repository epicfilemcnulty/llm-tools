import torch
from exllamav2.generator import ExLlamaV2Sampler
from transformers import GenerationConfig

def exl2_query(query, sampler, tokenizer, generator, lora, sc=[]):
 
    stop_conditions = [tokenizer.eos_token_id] + sc
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = sampler['temperature']
    settings.top_k = sampler['top_k']
    settings.top_p = sampler['top_p']
    settings.min_p = sampler['min_p']
    settings.token_repetition_penalty = sampler['repetition_penalty']

    input_ids = tokenizer.encode(query, add_bos = sampler['add_bos'], add_eos = sampler['add_eos'], encode_special_tokens = sampler['encode_special_tokens'])
    prompt_tokens = input_ids.shape[-1]
 
    generator.set_stop_conditions(stop_conditions)
    generator.begin_stream(input_ids, settings, loras = lora)
    generated_tokens = 0
    new_text = ""
    while True:
        chunk, eos, tokens = generator.stream()
        generated_tokens += 1
        new_text += chunk
        if eos or generated_tokens == sampler['max_new_tokens']:
            break

    stop_reason = "eos" if eos else "length"
    return new_text, prompt_tokens, generated_tokens, stop_reason

def tf_query(query, sampler, model, tokenizer, sc=[]):
 
    input_ids = tokenizer(query, return_tensors='pt').input_ids.to('cuda')
    prompt_tokens = len(input_ids[0])
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = sampler['max_new_tokens']
    gen_cfg.top_p = sampler['top_p']
    gen_cfg.top_k = sampler['top_k']
    gen_cfg.repetition_penalty = sampler['repetition_penalty']
    gen_cfg.temperature = sampler['temperature']
    gen_cfg.do_sample = True
    gen_cfg.num_beams = 1
    gen_cfg.num_return_sequences=1
    gen_cfg.remove_invalid_values=True
    outputs = model.generate(
        inputs=input_ids,
        generation_config = gen_cfg,
    )
    new_text = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    new_tokens = len(outputs[0]) - prompt_tokens
    return new_text, prompt_tokens, new_tokens
