# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Weight adapters for IBM SMI-TED ``model_weights.safetensors``."""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

# Training-only heads not used by the encode / property graphs.
_IGNORE_PREFIXES = (
    "encoder.lang_model.",
    "decoder.lang_model.",
    "decoder.autoencoder.decoder.",
)


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Filter Hub / finetune safetensors for the MAX SMI-TED graph.

    Keeps encode path weights and optional ``net.*`` property head.
    Drops LM heads and the autoencoder decoder.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if any(weight_name.startswith(prefix) for prefix in _IGNORE_PREFIXES):
            continue
        if not (
            weight_name.startswith("encoder.")
            or weight_name.startswith("decoder.autoencoder.encoder.")
            or weight_name.startswith("net.")
        ):
            continue
        if value is None:
            continue
        new_state_dict[weight_name] = value.data()

    return new_state_dict
