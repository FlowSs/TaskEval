#!/bin/bash
for mod in deepseek yi-coder qwen-coder gemma magicoder llama gpt deepseek-chat
do
    python clean_json_and_transform_to_jsonl.py -m "$mod" -d humanevalplus
    evalplus.sanitize --samples $(pwd)/../../../data/humanevalplus/raw/results_humanevalplus_"$mod".jsonl
    mv $(pwd)/../../../data/humanevalplus/raw/results_humanevalplus_"$mod"-sanitized.jsonl $(pwd)/../../../data/humanevalplus/raw/results_humanevalplus_"$mod".jsonl
    docker run --rm --pull=always -v $(pwd)/../../../data/humanevalplus/raw:/app ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/results_humanevalplus_"$mod".jsonl
    python back_to_prompt_types.py -m "$mod"
done
