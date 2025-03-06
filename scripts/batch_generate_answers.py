import gc

import pandas as pd
import torch

from tqdm import tqdm
from typing import List
from pandas import DataFrame
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import get_device, clear_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

def save_answers(checkpoint, output_dir, temperature, top_p, answers):
    print("Saved answers for checkpoint {}".format(checkpoint))
    temperature = str(temperature).replace(".", "_")
    top_p = str(top_p).replace(".", "_")
    pd.DataFrame({'answer': answers}).to_csv(f"{output_dir}/{checkpoint}-temp-{temperature}-top_p-{top_p}",index=False)

def get_assistant_answer(string: str) -> str:
    assistant_prefix = '<|im_start|>assistant\n'
    index = string.find(assistant_prefix)
    if index != -1:
        return string[index + len(assistant_prefix):].replace('<|im_end|>', '')
    else:
        print("Assistant string not found")
        return ''

def format_text(data, tokenizer):
    promts = []
    for el in data['chosen']:
        promt = [el[0]]
        formatted_promt = tokenizer.apply_chat_template(promt, add_generation_prompt=True, tokenize=False)
        promts.append(formatted_promt)
    return {'promt': promts}

def tokenize_function(rows, tokenizer, promt_max_length=1024):
    return tokenizer(rows["promt"],
                     padding="max_length",
                     truncation=True,
                     max_length=promt_max_length)


def generate(checkpoints:List[str],
             max_promt_len:int,
             max_generation_len:int,
             temperature:float=0.1,
             top_p:float=0.95,
             top_k:int=50,
             texts_batch_size:int=2):

    for checkpoint in tqdm(checkpoints):

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        ds = (load_dataset("trl-lib/ultrafeedback_binarized", split="test")
              .filter(lambda x: len(x['chosen'][0]['content']) < max_promt_len)
              .map(lambda x: format_text(x, tokenizer), remove_columns=['chosen', 'rejected', 'score_chosen', 'score_rejected'], batched=True)
              .map(lambda x: tokenize_function(x, tokenizer, max_promt_len), batched=True))

        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dl = DataLoader(ds, batch_size=texts_batch_size, shuffle=False)

        model = model.eval()
        model.to(get_device())
        answers = []

        with torch.inference_mode():

            for batch in dl:

                input_ids = batch["input_ids"].to(get_device())
                attention_mask = batch["attention_mask"].to(get_device())

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_generation_len,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                responses = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

                for response in responses:
                    answers.append(get_assistant_answer(response))

            try:
                del model
                del tokenizer
            except:
                pass

            gc.collect()
            clear_cache()

        save_answers(checkpoint, "data", temperature, top_p, answers)

if __name__ == "__main__":

    PROMT_MAX_LENGTH = 1024
    ASSISTANT_ANSWER_MAX_LENGTH = 50

    checkpoints = [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        # "mikheevshow/DPO-alpha-divergence-alpha_0_5_beta_0_1",
        # "mikheevshow/DPO-forward_kl_beta_0_1",
        # "mikheevshow/DPO-js_divergence_beta_0_1",
        # "mikheevshow/DPO-reverse_kl_beta_0_1",
        # "mikheevshow/DPO-reverse_kl_beta_5_0",
        # "mikheevshow/DPO-reverse_kl_beta_1_0",
        # "mikheevshow/DPO-reverse_kl_beta_0_05"
    ]

    generate(checkpoints, max_promt_len=PROMT_MAX_LENGTH, max_generation_len=ASSISTANT_ANSWER_MAX_LENGTH)


# Вероятности обученной модели
# Сунуть их в forward sft модели


