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


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
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
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2
        },
        "t5-large": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2
        }
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "t5-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-base/model_state.pdparams",
            "t5-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-large/model_state.pdparams",
        }
    }

    base_model_prefix = "fnet"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range if hasattr(
                                             self, "initializer_range") else
                                         self.fnet.config["initializer_range"],
                                         shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class FNetEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=4):
        super(FNetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=self.pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class FNetIntermediate(nn.Layer):
    pass
class FNetBasicFourierTransform(nn.Layer):
    pass
class FNetFourierTransform(nn.Layer):
    pass
class FNetLayer(nn.Layer):
    pass

class FNetEncoder(nn.Layer):
    pass

@register_base_model
class FNetModel(FNetPreTrainedModel):
    """
    The model can behave as an encoder, following the architecture described in `FNet: Mixing Tokens with Fourier
    Transforms <https://arxiv.org/abs/2105.03824>`__ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago
    Ontanon.
    """
    def __init__(self,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 hidden_act="gelu_new",
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=4,
                 initializer_range=0.02,
                 pad_token_id=3,
                 bos_token_id=1,
                 eos_token_id=2,
                 add_pooling_layer=True):
        super(FNetModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = FNetEmbeddings(vocab_size, hidden_size,
                                         hidden_dropout_prob,
                                         max_position_embeddings,
                                         type_vocab_size)
