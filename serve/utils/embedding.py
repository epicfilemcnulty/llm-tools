# Soon to be refactored...
import argparse
import numpy
import torch
import torch.nn.functional as F
from pgvector.psycopg2 import register_vector
import psycopg2
import bottle
from bottle import Bottle, run, route, request
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10

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

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Path to the model dir")
parser.add_argument('-d', '--database', required=True, type=str, help="Database name")
parser.add_argument('-u', '--db_user', required=True, type=str, help="Database user name")
parser.add_argument('--host', required=False, default="127.0.0.1", type=str, help="Database host")
parser.add_argument('-l', '--length', default=0, required=False, type=int, help="Context length")
parser.add_argument('--port', default=8014, required=False, type=int, help="Port to listen on")
parser.add_argument('--ip', default='127.0.0.1', required=False, type=str, help="IP to listen on")
args = parser.parse_args()

# Initialize model and cache
config = ExLlamaV2Config()
config.model_dir = args.model
config.prepare()

if args.length != 0:
    config.max_input_len = args.length

model = ExLlamaV2(config)
print("Loading model: " + args.model)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)

# Connect to your postgres DB                                                   
conn = psycopg2.connect(database=args.database, user=args.db_user, host=args.host)
register_vector(conn)

app = Bottle()

@app.route('/search', method='POST')
def complete():
    data = request.json
    limit = data.get('limit', 3)
    query = data.get('query')

    input_ids = tokenizer.encode(query, add_eos = False, add_bos = False, encode_special_tokens = False)
    tokens = input_ids.size()[-1] 
    if tokens > config.max_input_len:
        return bottle.HTTPResponse(status=501, body=f"Length exceeded: {tokens}")
    model_output = model.forward(input_ids, return_last_state = True)
    normalized = F.normalize(model_output[1], p=2, dim=-1)
    embedding = normalized.squeeze().cpu().numpy()
    matches = []

    with conn:
        with conn.cursor() as cur:
            cur.execute('SELECT page_id FROM wiki_embeddings ORDER BY embedding <-> %s LIMIT ' + str(limit), (embedding,))
            matches = cur.fetchall()
            # Extract the first element from each tuple                                 
            matches = [match[0] for match in matches]

    return {
        "matches": matches,
    }


@app.route('/embed', method='POST')
def complete():
    data = request.json
    content = data.get('content')
    page_id = data.get('page_id')
    
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM wiki_pages WHERE id = %s",  (page_id,))
            rows = cur.fetchall()
            content = rows[0][0]

    input_ids = tokenizer.encode(content, add_eos = False, add_bos = False, encode_special_tokens = False)
    tokens = input_ids.size()[-1] 
    if tokens > config.max_input_len:
        return bottle.HTTPResponse(status=501, body=f"Length exceeded: {tokens}")

    model_output = model.forward(input_ids, return_last_state = True)
    normalized = F.normalize(model_output[1], p=2, dim=-1)
    embedding = normalized.squeeze().cpu().numpy()

    # Connect to your postgres DB                                                   
    with conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO wiki_embeddings (page_id, embedding) VALUES (%s, %s)", (page_id, embedding))

    return {
        "status": "OK",
        "tokens": tokens,
    }

run(app, host=args.ip, port=args.port)
conn.close()
