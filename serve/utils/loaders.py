from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

def load_exl2_model(model_dir, context_length=None, lora_dir=None):
    # Initialize model and cache
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    if context_length is not None:
        config.max_seq_len = context_length

    model = ExLlamaV2(config)
    print("Loading model: " + model_dir)
    model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache_8bit(model, lazy = not model.loaded)
    lora = None
    if lora_dir is not None:
        lora = ExLlamaV2Lora.from_directory(model, lora_dir)
    # Initialize generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    # Make sure CUDA is initialized so we can measure performance
    generator.warmup()
    return { "model": model, "generator": generator, "tokenizer": tokenizer, "cache": cache, "lora": lora, "type": "exl2" }
