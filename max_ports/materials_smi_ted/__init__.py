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
"""IBM SMI-TED custom MAX architecture."""

from .arch import smi_ted_arch
from .model import SmiTedInputs, SmiTedPipelineModel
from .model_config import SmiTedModelConfig

ARCHITECTURES = [smi_ted_arch]

__all__ = [
    "ARCHITECTURES",
    "SmiTedInputs",
    "SmiTedModelConfig",
    "SmiTedPipelineModel",
    "smi_ted_arch",
]
