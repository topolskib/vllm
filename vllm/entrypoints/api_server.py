import argparse
import json
from typing import AsyncGenerator
import torch
import gc
import ray
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from pathlib import Path

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

def get_model_files(model_path):
    bin_files = list(model_path.glob("pytorch_model*.bin"))
    sft_files = list(model_path.glob("model*.safetensors"))
    return bin_files + sft_files

def get_model_path(model_path):
    model_path = Path(model_path)

    if len(get_model_files(model_path)) > 0:
        return str(model_path)
    merged_path = model_path / "merged"
    if len(get_model_files(merged_path)) > 0:
        return str(merged_path)
    raise ValueError("Incorrect model path")

@app.post("/change_model")
async def change_model(request: Request) -> Response:
    request_dict = await request.json()
    new_path = request_dict.pop("model_path")
    print(f"Request with new path: {new_path}")
    try:
        new_path = get_model_path(new_path)
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
 
    global engine_args
    current_path = engine_args.model
    if current_path == new_path:
        return Response(status_code=200)
    
    try:
        ray.shutdown()
        global engine
        del engine        
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()

        engine_args.model = new_path
        engine_args.tokenizer = new_path
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return Response(status_code=200)
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
        


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
