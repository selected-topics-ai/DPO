import gc
import argparse

from typing import List
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_device, seed_everything, clear_cache


def train(checkpoint:str="HuggingFaceTB/SmolLM-135M-Instruct",
          dataset:str="trl-lib/ultrafeedback_binarized",
          output_dir:str="./SMOL_DPO_REVERSE_KL",
          betas: List[float] = frozenset([0.05, 0.1, 1.0, 5.0]),
          bf16:bool=True,
          max_steps:int=200,
          learning_rate:float=5e-5,
          per_device_train_batch_size:int=4,
          per_device_eval_batch_size:int=4,
          gradient_accumulation_steps:int=4,
          max_length:int=1536,
          max_prompt_length:int=1024,
          logging_steps:int=1,
          optimizer:str= "paged_adamw_8bit" if get_device() == "cuda" else "adamw_torch"
          ):

    print("Prepare dataset...")

    ds = load_dataset(dataset)
    train_prefs = ds["train"]
    test_prefs = ds["test"]

    print("Prepare model...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    for beta in betas:

        gc.collect()
        clear_cache()

        pi_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(get_device())
        pi_model.config.use_cache = False

        args = DPOConfig(
            learning_rate=learning_rate,
            max_steps=max_steps,
            output_dir=output_dir + "_" + str(beta).replace(".", "_"),
            bf16=bf16,
            beta=beta,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            optim=optimizer,
        )

        trainer = DPOTrainer(
            model=pi_model,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_prefs,
            eval_dataset=test_prefs,
        )

        trainer.train()



if __name__ == '__main__':

    parser = argparse.ArgumentParser("TRAIN_DPO_REVERSE_KL_DIVERGENCE")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./SMOL_DPO_REVERSE_KL")

    args = parser.parse_args()

    print("Training DPO Reverse KL Divergence")
    print(f"Args: {args}")

    seed_everything(args.seed)

    train(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
    )