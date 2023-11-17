# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class CoregPromptTuningInit(str, enum.Enum):
    # initialize prompt with text
    TEXT = "TEXT"
    # initialize prompt with random matrix
    RANDOM = "RANDOM"



@dataclass
class CoregPromptTuningConfig(PromptLearningConfig):
    prompt_tuning_init: Union[CoregPromptTuningInit, str] = field(
        default=CoregPromptTuningInit.RANDOM,
        metadata={
            "help": (
                "How to initialize the prompt tuning parameters. Can be one of TEXT, RANDOM"
            ),
        },
    )
    prompt_tuning_init_state_dict_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path of source state dict. This is required when training the downstream target prompt from "
                "the pretrained source prompt"
            ),
        },
    )
    prompt_tuning_init_task: Optional[int] = field(default=0, metadata={"help": "source task id for initialization"})
    attention_dim: Optional[int] = field(default=1024, metadata={"help": "attention dim for square K Q V matrices"})
    num_views: Optional[int] = field(default=2, metadata={"help": "number of views"})
    decorrelate: Optional[bool] = field(default=False, metadata={"help": "Whether to decorrelate attention output"})
    decorrelate_lambda: Optional[float] = field(default=0.2, metadata={"help": "How much to decorrelate attention output"})

    def __post_init__(self):
        self.peft_type = PeftType.COREG_PROMPT_TUNING
