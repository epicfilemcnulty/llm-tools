import argparse
import time
import torch
import uuid
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from bottle import Bottle, run, route, request

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Grasping Model")
parser.add_argument('-a', '--model_name', required=False, type=str, default="uknown", help="Grasping Model's Alias")
parser.add_argument('-l', '--lora_dir', required=False, type=str, default='', help="Path to lora directory")
parser.add_argument('-r', '--remote', action = "store_true", help="Trust remote code (default is False)")
parser.add_argument('-f', '--flash_attn', action = "store_true", help="Use flash-attention (default is False)")
parser.add_argument('--port', default=8013, required=False, type=int, help="Port to listen on")
parser.add_argument('--ip', default='127.0.0.1', required=False, type=str, help="IP to listen on")
args = parser.parse_args()

app = Bottle()

def load_model():
    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=nf4_config, max_memory=max_memory, trust_remote_code=args.remote, use_flash_attention_2=args.flash_attn)
    print(model.generation_config)

    if args.lora_dir != '':
        model = PeftModel.from_pretrained(model, args.lora_dir)

    return model, tokenizer

llm, tokenizer = load_model()

conversations = {}

def full_conversation(idx):
    chat = ''
    for message in conversations[idx]['messages']:
        if message['role'] == 'system':
            chat += message['content']
        if message['role'] == 'user':
            chat += conversations[idx]['prefix'] + message['content'] + conversations[idx]['infix']
        if message['role'] == 'assistant':
            chat += conversations[idx]['suffix'] + message['content'] + '\n'
 
    if conversations[idx]['messages'][-1]['role'] == 'user':
        chat += conversations[idx]['suffix']
 
    return chat
 
@app.route('/prompt', method='PUT')
def set_prompt():
    data = request.json
    conversation_uuid = data.get('uuid', str(uuid.uuid4()))
    messages = data.get('messages', [{'role':'system', 'content':''}])
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    infix = data.get('infix', '')
    conversations[conversation_uuid] = {
        "messages": messages,
        "prefix": prefix,
        "suffix": suffix,
        "infix": infix
    }
    return {"message": "Prompt set", "uuid": conversation_uuid}

@app.route('/chat', method='POST')
def chat():
    data = request.json
    conversation_uuid = data['uuid']
    if conversation_uuid not in conversations:
        return {"uuid":conversation_uuid, "message": "not found"}
 
    temperature = data.get('temperature', 0.5)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.75)
    repetition_penalty = data.get('repetition_penalty', 1.05)
    max_new_tokens = data.get('max_length', 256)
    query = data.get('query')
 
    conversations[conversation_uuid]['messages'].append({'role':'user','content':query})
    full_ctx = full_conversation(conversation_uuid)

    start_time = time.time_ns()
    input_ids = tokenizer(full_ctx, return_tensors="pt").input_ids.to('cuda')
    outputs = llm.generate(
        inputs=input_ids,
        do_sample=True, 
        num_beams=1,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        repetition_penalty = repetition_penalty,
        num_return_sequences=1,
        remove_invalid_values=True,
    )
    answer = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    conversations[conversation_uuid]['messages'].append({'role':'assistant','content':answer})
    new_tokens = len(outputs[0]) - len(input_ids[0])
    end_time = time.time_ns()
    secs = (end_time - start_time) / 1e9
    return {
        "uuid": conversation_uuid,
        "text": answer,
        "ctx": len(outputs[0]),
        "tokens": new_tokens,
        "rate": new_tokens / secs,
        "model": args.model_name,
        "backend": "transformers",
    }

@app.route('/complete', method='POST')
def complete():
    data = request.json
 
    temperature = data.get('temperature', 0.5)
    add_bos = data.get('add_bos', False)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.9)
    repetition_penalty = data.get('repetition_penalty', 1.05)
    max_new_tokens = data.get('max_length', 512)
    query = data.get('query')

    tok = AutoTokenizer.from_pretrained(args.model, add_bos_token=add_bos)
    start_time = time.time_ns()
    input_ids = tok(query, return_tensors="pt").input_ids.to('cuda')
    outputs = llm.generate(
        inputs=input_ids,
        do_sample=True, 
        num_beams=1,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        repetition_penalty = repetition_penalty,
        num_return_sequences=1,
        remove_invalid_values=True,
    )
    answer = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:])[0]
    new_tokens = len(outputs[0]) - len(input_ids[0])
    end_time = time.time_ns()
    secs = (end_time - start_time) / 1e9
    return {
        "text": answer,
        "ctx": len(outputs[0]),
        "tokens": new_tokens,
        "rate": new_tokens / secs,
        "model": args.model_name,
        "backend": "transformers",
    }

run(app, host=args.ip, port=args.port)
