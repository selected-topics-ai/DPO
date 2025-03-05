import gc
import pandas as pd

from tqdm import tqdm
from typing import List
from datasets import load_dataset
from utils import get_device, clear_cache
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_assistant_answer(string: str) -> str:
    assistant_prefix = '<|im_start|>assistant\n'
    index = string.find(assistant_prefix)
    if index != -1:
        return string[index + len(assistant_prefix):].replace('<|im_end|>', '')
    else:
        print("Assistant string not found")
        return ''

def generate(checkpoints: List[str],
             output_dir: str = "../data",
             generate_times: int = 3,
             n_promts: int = 100,
             promt_max_len: int = 1024,
             max_new_tokens: int = 512,
             temperature: float = 1.0,
             top_p: float = 0.95,
             dataset_name: str = "trl-lib/ultrafeedback_binarized",
             device=get_device()):

    ds = load_dataset(dataset_name)
    test_prefs = ds["test"]

    promts_dataset = (test_prefs
                   .filter(lambda x: len(x['chosen'][0]['content']) < promt_max_len)
                   .map(lambda x: {'promt': [x['chosen'][0]]},
                        remove_columns=['chosen', 'rejected', 'score_chosen', 'score_rejected'])
                   .select(indices=[i for i in range(n_promts)]))


    for checkpoint in checkpoints:

        print("Generating answers for checkpoint {}".format(checkpoint))

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        model.config.use_cache = False

        for time in range(generate_times):

            print("Generating answers for checkpoint {}, iteration {}".format(checkpoint, time))

            answers = []

            for promts_dict in tqdm(promts_dataset):
                promt = promts_dict['promt']
                input_text = tokenizer.apply_chat_template(promt, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
                outputs = model.generate(inputs,
                                         max_new_tokens=max_new_tokens,
                                         temperature=temperature,
                                         do_sample=True,
                                         top_p=top_p)
                output_tokens = tokenizer.decode(outputs[0])
                answer = get_assistant_answer(output_tokens)
                answers.append(answer)

            print("Saved answers for checkpoint {}, iteration {}".format(checkpoint, time))
            pd.DataFrame({'answer': answers}).to_csv(f"{output_dir}/{checkpoint}_generation_{time}.csv", index=False)

        try:
            del tokenizer
            del model
        except:
            pass

        gc.collect()
        clear_cache()

if __name__ == "__main__":

    checkpoints = [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "mikheevshow/DPO-alpha-divergence-alpha_0_5_beta_0_1",
        "mikheevshow/DPO-forward_kl_beta_0_1",
        "mikheevshow/DPO-js_divergence_beta_0_1",
        "mikheevshow/DPO-reverse_kl_beta_0_1",
        "mikheevshow/DPO-reverse_kl_beta_5_0",
        "mikheevshow/DPO-reverse_kl_beta_1_0",
        "mikheevshow/DPO-reverse_kl_beta_0_05"
    ]

    generate(checkpoints)