# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

import math

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F

from ..model_utils import PretrainedModel, register_base_model

__all__ = [
    'FnetModel',
    "FnetPretrainedModel",
]


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class FNetPreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained FNet models. It provides FNet related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "fnet"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "fnet-base": {
            "actual_seq_length": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "fnet",
            "num_hidden_layers": 12,
            "pad_token_id": 3,
            "type_vocab_size": 4,
            "use_fft": True,
            "vocab_size": 32000
        },
        "t5-large": {}
    }
    