import argparse
from typing import List
from utils import get_device
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import os
import pandas as pd

import llm_blender

def get_assistant_answer(string: str) -> str:
    assistant_prefix = '<|im_start|>assistant\n'
    index = string.find(assistant_prefix)
    if index != -1:
        return (string[index + len(assistant_prefix):]
                .replace('<|im_end|>', '')
                .replace('<|endoftext|>', '')
                .replace('<|im_start|>', ''))
    else:
        print("Assistant answer not found. Check tokenizer's chat template")
        return ''


def generate_answer(messages: List[any], 
                    tokenizer: PreTrainedTokenizerFast, 
                    model, device=get_device(),
                    top_p: float=0.95,
                    temperature: float=1.0,
                    max_new_tokens: int=512) -> str:
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             do_sample=True,
                             top_p=top_p)
    output_tokens = tokenizer.decode(outputs[0])
    return get_assistant_answer(output_tokens)




def calc_winrate(promts: List[str], aligned_answers: List[str], base_answers: List[str]) -> float:

    inputs = promts
    candidates = []

    if len(promts) != len(aligned_answers) or len(promts) != len(base_answers):
        raise RuntimeError("Promts, answers list lengths do not match")

    for aligned_answer, base_answer in zip(aligned_answers, base_answers):
        candidates.append([aligned_answer, base_answer])
                        
    ranks = blender.rank(inputs, candidates, batch_size=1)

    aligned_answer_win_count = 0
    for rank in ranks:
        if rank[0] == 1:
            aligned_answer_win_count += 1

    return float(aligned_answer_win_count) / len(aligned_answers)


if __name__ == "__main__":

    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM", device=str(get_device()))

    ds = (load_dataset("trl-lib/ultrafeedback_binarized", split="test")
          .filter(lambda x: len(x['chosen'][0]['content']) < 1024)
          .select(indices=[i for i in range(100)]))

    prompts = []

    for row in ds['chosen']:
        prompts.append(row[0]['content'])

    aligned_models_winrate = {}

    sft_answers = pd.read_csv('data/HuggingFaceTB/SmolLM2-135M-Instruct-temp-0_8-top_p-0_95')

    for path in os.walk('./data/mikheevshow'):
        for filename in path[2]:
            filepath = path[0] + '/' + filename
            aligned_model_ds = pd.read_csv(filepath)
            wr = calc_winrate(prompts, aligned_model_ds.fillna('')['answer'], sft_answers.fillna('')['answer'])
            aligned_models_winrate[filename] = wr
            print(filename, wr)

    pd.DataFrame(aligned_models_winrate).melt().to_csv('aligned_models_winrate.csv')