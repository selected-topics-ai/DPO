import argparse

from utils import get_device, seed_everything
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_config import FDivergenceType
from transformers import AutoModelForCausalLM, AutoTokenizer


def train(checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct",
          output_dir:str="./SMOL_DPO_JS_DIVERGENCE",
          dataset:str="trl-lib/ultrafeedback_binarized",
          beta:float=0.1,
          bf16:bool=True,
          max_steps:int=200,
          logging_steps:int=1,
          learning_rate: float = 5e-5,
          max_length:int= 1536,
          max_prompt_length:int= 1024,
          per_device_eval_batch_size:int=4,
          per_device_train_batch_size:int=4,
          gradient_accumulation_steps:int=4,
          optimizer:str="paged_adamw_8bit" if get_device() == "cuda" else "adamw_torch",
          f_alpha_divergence_coef:float=0.5,
          device=get_device(),
          ):

    ds = load_dataset(dataset)
    train_prefs = ds["train"]
    test_prefs = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    pi_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    pi_model.config.use_cache = False

    divergence = FDivergenceType.JS_DIVERGENCE

    args = DPOConfig(
        learning_rate=learning_rate,
        max_steps=max_steps,
        output_dir=output_dir,
        bf16=bf16,
        beta=beta,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        optim=optimizer,
        f_divergence_type=divergence,
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

    parser = argparse.ArgumentParser("TRAIN_DPO_ALPHA_DIVERGENCE")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_coef", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./SMOL_DPO_ALPHA_KL")

    args = parser.parse_args()

    print("Training DPO Alpha Divergence")
    print(f"Args: {args}")

    seed_everything(args.seed)

    train(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        f_alpha_divergence_coef=args.alpha_coef,
    )