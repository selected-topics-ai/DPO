import gc
import os

import pandas as pd
import llm_blender

from tqdm import tqdm
from datasets import load_dataset
from utils import get_device, clear_cache

if __name__ == "__main__":

    dataset_name = "trl-lib/ultrafeedback_binarized"
    promt_max_len = 1024
    n_promts = 100

    sft_anwers = [
        "../data/HuggingFaceTB-SmolLM-135M-Instruct_generation_0.csv"
    ]

    corresponding_dpo_models_answers = [
            # "../data/mikheevshow-DPO-alpha-divergence-alpha_0_5_beta_0_1_generation_0.csv",
            # "../data/mikheevshow-DPO-forward_kl_beta_0_1_generation_0.csv",
            # "../data/mikheevshow-DPO-js_divergence_beta_0_1_generation_0.csv",
            # "../data/mikheevshow-DPO-reverse_kl_beta_0_1_generation_0.csv",
            # "../data/mikheevshow-DPO-reverse_kl_beta_1_0_generation_0.csv",
            # "../data/mikheevshow-DPO-reverse_kl_beta_5_0_generation_0.csv",
            "../data/mikheevshow-DPO-reverse_kl_beta_0_05_generation_0.csv"
    ]


    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM", device=str(get_device()))

    ds = load_dataset(dataset_name)
    test_prefs = ds["test"]

    promts_dataset = (test_prefs
                      .filter(lambda x: len(x['chosen'][0]['content']) < promt_max_len)
                      .map(lambda x: {'promt': [x['chosen'][0]]},
                           remove_columns=['chosen', 'rejected', 'score_chosen', 'score_rejected'])
                      .select(indices=[i for i in range(n_promts)]))

    out_df = pd.DataFrame({'model': [], 'winrate': []})

    for i, path_sft in enumerate(sft_anwers):
        df_sft = pd.read_csv(path_sft)

        for j, path_dpo in enumerate(tqdm(corresponding_dpo_models_answers)):
            df_dpo = pd.read_csv(path_dpo)

            inputs = []
            candidates = []
            for promt_dict, df_dpo_answ, df_sft_answ in zip(promts_dataset, df_dpo['answer'].tolist(), df_sft['answer'].tolist()):
                inputs.append(promt_dict["promt"][0]['content'])
                candidates.append([df_dpo_answ, df_sft_answ])

            ranks = blender.rank(inputs, candidates, batch_size=1)

            aligned_answer_win_count = 0
            for rank in ranks:
                if rank[0] == 1:
                    aligned_answer_win_count += 1

            winrate = float(aligned_answer_win_count) / len(inputs)

            new_row = pd.DataFrame({'model': [path_dpo.replace('/data', '')], 'winrate': [winrate]})
            out_df = pd.concat([out_df, new_row], ignore_index=True)

    out_df.to_csv('data/winrate.csv', index=False)