import argparse
import sys, os
import uuid
import time
import bottle
from bottle import Bottle, run, route, request
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10

from utils.loaders import load_exl2_model
from utils.generation import exl2_query, tf_query

parser = argparse.ArgumentParser()
parser.add_argument('--port', default=8013, required=False, type=int, help="Port to listen on")
parser.add_argument('--ip', default='127.0.0.1', required=False, type=str, help="IP to listen on")
args = parser.parse_args()

models = {}
conversations = {}
app = Bottle()

@app.route('/load', method='POST')
def load_model():
    data = request.json
    model_dir = data.get('model_dir')
    model_type = data.get('model_type')
    model_alias = data.get('model_alias')
    if not model_dir:
        return {"error": "model_dir is required"}
    if not model_type:
        return {"error": "model_type is required"}
    if not model_alias:
        return {"error": "model_alias is required"}
    context_length = data.get('context_length')
    lora_dir = data.get('lora_dir')
    if model_type == "exl2":
        models[model_alias] = load_exl2_model(model_dir, context_length, lora_dir)
        return {"message": "model loaded"}
    if model_type == "tf":
        models[model_alias] = load_tf_model(model_dir, context_length, lora_dir)
        return {"message": "model loaded"}

@app.route('/models', method='GET')
def loaded_models():
    return { models.keys() }

@app.route('/chat', method='POST')
def chat():
    data = request.json
    conversation_uuid = data['uuid']
    if conversation_uuid not in conversations:
        return {"uuid":conversation_uuid, "message": "not found"}
 
    temperature = data.get('temperature', 0.5)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.75)
    min_p = data.get('min_p', 0.0)
    add_bos = data.get('add_bos', True)
    add_eos = data.get('add_eos', False)
    encode_special_tokens = False
    repetition_penalty = data.get('repetition_penalty', 1.05)
    max_new_tokens = data.get('max_length', 256)

    query = data.get('query')

    model_alias = data.get('model')
    model_type = models[model_alias]["type"]

    sampler = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "add_bos": add_bos,
    }

    conversations[conversation_uuid]['messages'].append({'role':'user','content':query})
    full_ctx = full_conversation(conversation_uuid)

    start_time = time.time_ns()
    if model_type == "exl2":
        new_text, prompt_tokens, generated_tokens, stop_reason = exl2_query(full_ctx, sampler, models[model_alias]["tokenizer"], models[model_alias]["generator"], models[model_alias]["lora"])
    if model_type == "tf":
        new_text, prompt_tokens, generated_tokens, stop_reason = tf_query(full_ctx, sampler)
    end_time = time.time_ns()

    secs = (end_time - start_time) / 1e9
    conversations[conversation_uuid]['messages'].append({'role':'assistant','content':new_text})

    return {
        "uuid": conversation_uuid,
        "text": new_text,
        "tokens": generated_tokens,
        "rate": generated_tokens / secs,
        "model": model_alias,
        "backend" : model_type,
        "stop": stop_reason,
        "ctx" : prompt_tokens + generated_tokens
    }
