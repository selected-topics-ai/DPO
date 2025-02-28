import torch

from torch import nn
from typing import Callable, Optional, Union
from trl.trainer.utils import cap_exp
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

import torch.nn.functional as F


class ForwardKLDPOTrainer(DPOTrainer):
    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 args: Optional[DPOConfig] = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
                 processing_class: Optional[
                    Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
                ] = None,
                model_init: Optional[Callable[[], PreTrainedModel]] = None,
                compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
                callbacks: Optional[list[TrainerCallback]] = None,
                optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                peft_config: Optional[dict] = None,):
        super().__init__(model=model,
                         ref_model=ref_model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         processing_class=processing_class,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         peft_config=peft_config)

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        device = self.accelerator.device

        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        logits = (1 / cap_exp(rejected_logratios)) - (1 / cap_exp(chosen_logratios))

        losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards