import torch
import typing as t
import transformers
  
def _nearest_divisible(num: int, divisor: int) -> int:
    '''Returns the nearest number to `num` that is divisible by `divisor`.'''
    return (num + divisor - 1) // divisor * divisor

def add_special_tokens(
    special_tokens: t.Dict[str, t.Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    tokenizer.add_special_tokens(special_tokens)

    # Size is rounded up to the nearest number divisible by 64 for performance
    # reasons.
    new_size = _nearest_divisible(num=len(tokenizer), divisor=64)
    old_size = model.config.vocab_size

    if new_size == old_size:
        # No resizing needs to be done, let's bail!
        return

    # Need to resize the token embeddings. We initialize the new positions with
    # the mean of the existing ones to cut down on required training time.
    model.resize_token_embeddings(new_size)
    new_positions_count = new_size - old_size

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # This is just to keep the LSP happy.
    assert isinstance(input_embeddings, torch.Tensor)
    assert isinstance(output_embeddings, torch.Tensor)

    input_embeddings_avg = input_embeddings[:-new_positions_count].mean(dim=0,
                                                             keepdim=True)
    output_embeddings_avg = output_embeddings[:-new_positions_count].mean(dim=0,
                                                               keepdim=True)

    input_embeddings[-new_positions_count:] = input_embeddings_avg
    output_embeddings[-new_positions_count:] = output_embeddings_avg
