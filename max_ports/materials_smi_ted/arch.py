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
"""Architecture registration for IBM SMI-TED."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import SmiTedBatchProcessor
from .model import SmiTedInputs, SmiTedPipelineModel
from .model_config import SmiTedModelConfig
from .tokenizer import SmiTedTokenizer

smi_ted_arch = SupportedArchitecture(
    name="SmiTedModel",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "ibm-research/materials.smi-ted",
    ],
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=SmiTedPipelineModel,
    tokenizer=SmiTedTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=SmiTedModelConfig,
    batching=SmiTedBatchProcessor,
)
