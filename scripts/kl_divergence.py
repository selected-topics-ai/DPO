import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_kl_divergence(aligned_model: nn.Module,
                          ref_model: nn.Module,
                          tokenizer: PreTrainedTokenizer,
                          prompt: str,
                          answer: str):

    full_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    inputs = tokenizer(full_input, return_tensors="pt")

    assistant_response = f"<|im_start|>assistant\n{answer}<|im_end|>"
    assistant_ids = tokenizer(assistant_response, return_tensors="pt").input_ids


    with torch.no_grad():
        outputs_model = aligned_model(**inputs).logits
        outputs_ref = ref_model(**inputs).logits

        assistant_start_idx = inputs.input_ids.shape[1] - assistant_ids.shape[1]

        logits_model_assistant = outputs_model[:, assistant_start_idx+4:-1, :]

        logits_ref_assistant = outputs_ref[:, assistant_start_idx+4:-1, :]

        logp_model = F.log_softmax(logits_model_assistant, dim=-1)
        logp_ref = F.log_softmax(logits_ref_assistant, dim=-1)

        kl_div = F.kl_div(logp_model, logp_ref.exp(), reduction="batchmean", log_target=False)

        return kl_div.item()



def calculate_mean_kl(aligned_model: nn.Module, ref_model: nn.Module, tokenizer: PreTrainedTokenizer, generated_answers: pd.Series):

    ds = (load_dataset("trl-lib/ultrafeedback_binarized", split="test")
          .filter(lambda x: len(x['chosen'][0]['content']) < 1024)
          .select(indices=[i for i in range(100)]))

    prompts = []

    for row in ds['chosen']:
        prompts.append(row[0]['content'])

    kls = []
    for prompt, answer in zip(prompts, generated_answers):
        kls.append(compute_kl_divergence(aligned_model, ref_model, tokenizer, prompt, answer))

    return kls


if __name__ == "__main__":

    sft_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    mean_kls = {}

    for path in os.listdir('../data/answers/aligned'):
        model_name = path[:path.find('-temp')]

        tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
        sft = AutoModelForCausalLM.from_pretrained(sft_model_name)
        aligned_model = AutoModelForCausalLM.from_pretrained('mikheevshow/' + model_name)

        answers = pd.read_csv(f'../data/answers/aligned/{path}')['answer']

        mean_kls[model_name] = calculate_mean_kl(aligned_model, sft, tokenizer, answers)