from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    file_path: Optional[str] = field(
        default='/home/yeqin/data/2024-07-02_train.pt',#'/root/autodl-tmp/hotpotqa',
    )
    doc_path: Optional[str] = field(
        default='/home/yeqin/basic_docstring',#'/root/autodl-tmp/hotpotqa',
    )
    max_length: int = field(default=512)
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/yeqin/model/bge-base-zh-v1.5",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )

@dataclass
class TrainingArguments:
    weight_decay: float = field(default=0.01)
    lr: float = field(default=2e-5)
    deepspeed:Optional[str] = field(
        default="./deepspeed.json",
        metadata={
            "help": "Deepspeed file path."
        }
    )
    local_rank: Optional[int] = field(default=0)
    train_epoch: Optional[int] = field(default=1)
    lr_scheduler_type: str = field(default="linear")
    train_batch_size: Optional[int] = field(default=32)
    num_warmup_steps: int = field(default=1000)
    num_epochs: int = field(default=1)

