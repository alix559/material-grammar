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
"""MAX graph for IBM SMI-TED (SMILES encoder-decoder foundation model).

Implements the inference ``encode()`` path from IBM/materials ``load.py``:
token embedding → 12-layer linear-attention encoder with rotary embeddings →
padded token tensor → decoder autoencoder encoder → 768-d SMILES embedding.

When ``config.huggingface_config.smi_ted_output == "property"``, also runs the
IBM ``Net`` head and returns a ``[batch, n_output]`` property prediction
(served via the embeddings API as a length-1 vector).
"""

from __future__ import annotations

from collections.abc import Mapping

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.weights import WeightData
from max.nn.activation import activation_function_from_name
from max.nn.embedding import Embedding
from max.nn.layer import Module
from max.nn.sequential import Sequential
from max.nn.linear import Linear
from max.nn.norm import LayerNorm

from .model_config import SmiTedModelConfig


def _reshape_heads(x: TensorValue, n_heads: int, head_dim: int) -> TensorValue:
    batch, seq_len, _ = x.shape
    return ops.reshape(x, (batch, seq_len, n_heads, head_dim))


def _merge_heads(x: TensorValue, n_heads: int, head_dim: int) -> TensorValue:
    batch, seq_len, _, _ = x.shape
    return ops.reshape(x, (batch, seq_len, n_heads * head_dim))


def _rotate_half(x: TensorValue) -> TensorValue:
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1, x2 = ops.split(x, [half, half], axis=-1)
    return ops.concat([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: TensorValue, k: TensorValue, inv_freq: TensorValue
) -> tuple[TensorValue, TensorValue]:
    seq_len = q.shape[1]
    device = q.device

    t = ops.range(0, seq_len, dtype=DType.float32, device=DeviceRef.CPU())
    t = t.to(device)
    freqs = ops.outer(t, inv_freq)
    emb = ops.concat([freqs, freqs], axis=-1)
    cos = ops.cos(emb)
    sin = ops.sin(emb)
    cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 2)
    sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 2)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class FeatureMap(Module):
    """ReLU generalized random features (Performer-style)."""

    def __init__(self, head_dim: int, num_feats: int, device: DeviceRef) -> None:
        self.omega = Weight(
            "omega",
            DType.float32,
            [head_dim, num_feats],
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: [B, L, H, D]; omega: [D, M] — match HF `x @ omega` on the last dim.
        # Do not reshape to (B*H, L, D): that scrambles the head layout.
        return ops.relu(x @ self.omega)


class InnerLinearAttention(Module):
    """Matches ``fast_transformers.attention.linear_attention.LinearAttention``."""

    def __init__(self, head_dim: int, num_feats: int, device: DeviceRef, eps: float = 1e-6) -> None:
        self.eps = eps
        self.feature_map = FeatureMap(head_dim, num_feats, device)

    def __call__(
        self,
        queries: TensorValue,
        keys: TensorValue,
        values: TensorValue,
        length_mask: TensorValue,
    ) -> TensorValue:
        phi_q = self.feature_map(queries)
        phi_k = self.feature_map(keys)
        mask = ops.reshape(
            length_mask,
            (length_mask.shape[0], length_mask.shape[1], 1, 1),
        )
        phi_k = phi_k * mask
        values = values * mask

        # phi_k: [B, L, H, M], values: [B, L, H, D]
        # Match fast_transformers: KV = einsum("nshd,nshm->nhmd", K, V)
        phi_k_h = ops.permute(phi_k, [0, 2, 3, 1])  # [B, H, M, L]
        values_h = ops.permute(values, [0, 2, 1, 3])  # [B, H, L, D]
        kv = phi_k_h @ values_h  # [B, H, M, D]

        phi_q_h = ops.permute(phi_q, [0, 2, 1, 3])  # [B, H, L, M]
        out_h = phi_q_h @ kv  # [B, H, L, D]

        # Z = 1 / (einsum("nlhd,nhd->nlh", Q, K.sum(1)) + eps)
        # ops.sum keeps the reduced axis as size 1.
        k_sum = ops.sum(phi_k, axis=1)  # [B, 1, H, M]
        z_denom = ops.sum(phi_q * k_sum, axis=-1) + self.eps  # [B, L, H, 1]
        z = ops.div(1.0, z_denom)
        out = ops.permute(out_h, [0, 2, 1, 3])  # [B, L, H, D]
        return out * z


class RotaryEmbedding(Module):
    def __init__(self, head_dim: int, device: DeviceRef) -> None:
        self.inv_freq = Weight(
            "inv_freq",
            DType.float32,
            [head_dim // 2],
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return x


class RotateAttentionLayer(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.n_heads = hf.n_head
        self.head_dim = hf.n_embd // hf.n_head
        inner_dim = self.n_heads * self.head_dim

        self.query_projection = Linear(
            hf.n_embd, inner_dim, dtype, device, has_bias=True
        )
        self.key_projection = Linear(
            hf.n_embd, inner_dim, dtype, device, has_bias=True
        )
        self.value_projection = Linear(
            hf.n_embd, inner_dim, dtype, device, has_bias=True
        )
        self.out_projection = Linear(
            inner_dim, hf.n_embd, dtype, device, has_bias=True
        )
        self.inner_attention = InnerLinearAttention(
            self.head_dim, hf.num_feats, device
        )
        self.rotaryemb = RotaryEmbedding(self.head_dim, device)

    def __call__(
        self, hidden_states: TensorValue, length_mask: TensorValue
    ) -> TensorValue:
        q = _reshape_heads(
            self.query_projection(hidden_states), self.n_heads, self.head_dim
        )
        k = _reshape_heads(
            self.key_projection(hidden_states), self.n_heads, self.head_dim
        )
        v = _reshape_heads(
            self.value_projection(hidden_states), self.n_heads, self.head_dim
        )
        q, k = _apply_rotary_pos_emb(q, k, self.rotaryemb.inv_freq)
        attn = self.inner_attention(q, k, v, length_mask)
        attn = _merge_heads(attn, self.n_heads, self.head_dim)
        return self.out_projection(attn)


class TransformerEncoderLayer(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.attention = RotateAttentionLayer(config)
        self.linear1 = Linear(hf.n_embd, hf.n_embd, dtype, device, has_bias=True)
        self.linear2 = Linear(hf.n_embd, hf.n_embd, dtype, device, has_bias=True)
        self.norm1 = LayerNorm(
            hf.n_embd, devices=[device], dtype=DType.float32, eps=1e-5, use_bias=True
        )
        self.norm2 = LayerNorm(
            hf.n_embd, devices=[device], dtype=DType.float32, eps=1e-5, use_bias=True
        )

    def __call__(
        self, hidden_states: TensorValue, length_mask: TensorValue
    ) -> TensorValue:
        attn_out = self.attention(hidden_states, length_mask)
        # Match fast_transformers: y = x = norm1(x + attn); return norm2(x + ffn(y))
        hidden_states = self.norm1(hidden_states + attn_out)
        gelu = activation_function_from_name("gelu")
        y = gelu(self.linear1(hidden_states))
        y = self.linear2(y)
        return self.norm2(hidden_states + y)


class TransformerEncoderBlocks(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        device = config.device
        self.layers = Sequential(
            [
                TransformerEncoderLayer(config)
                for _ in range(config.huggingface_config.n_layer)
            ]
        )
        self.norm = LayerNorm(
            hf.n_embd, devices=[device], dtype=DType.float32, eps=1e-5, use_bias=True
        )

    def __call__(
        self, hidden_states: TensorValue, length_mask: TensorValue
    ) -> TensorValue:
        for layer in self.layers.layers:
            hidden_states = layer(hidden_states, length_mask)
        return self.norm(hidden_states)


class MoLEncoder(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        self.tok_emb = Embedding(hf.vocab_size, hf.n_embd, config.dtype, config.device)
        self.blocks = TransformerEncoderBlocks(config)

    def __call__(
        self, input_ids: TensorValue, length_mask: TensorValue
    ) -> TensorValue:
        x = self.tok_emb(input_ids)
        return self.blocks(x, length_mask)


class AutoEncoderEncoder(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        dtype = config.dtype
        device = config.device
        feature_size = hf.max_len * hf.n_embd
        self.fc1 = Linear(feature_size, hf.n_embd, dtype, device, has_bias=True)
        self.ln_f = LayerNorm(
            hf.n_embd, devices=[device], dtype=DType.float32, eps=1e-5, use_bias=True
        )
        self.lat = Linear(hf.n_embd, hf.n_embd, dtype, device, has_bias=False)

    def __call__(self, flat_tokens: TensorValue) -> TensorValue:
        gelu = activation_function_from_name("gelu")
        x = gelu(self.fc1(flat_tokens))
        x = self.ln_f(x)
        return self.lat(x)


class AutoEncoder(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        self.encoder = AutoEncoderEncoder(config)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.encoder(x)


class MoLDecoder(Module):
    def __init__(self, config: SmiTedModelConfig) -> None:
        self.autoencoder = AutoEncoder(config)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.autoencoder.encoder(x)


class Net(Module):
    """IBM SMI-TED property head (inference: no dropout)."""

    def __init__(self, config: SmiTedModelConfig) -> None:
        hf = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.fc1 = Linear(hf.n_embd, hf.n_embd, dtype, device, has_bias=True)
        self.fc2 = Linear(hf.n_embd, hf.n_embd, dtype, device, has_bias=True)
        self.final = Linear(hf.n_embd, hf.n_output, dtype, device, has_bias=True)

    def __call__(self, embeddings: TensorValue) -> TensorValue:
        gelu = activation_function_from_name("gelu")
        x_out = gelu(self.fc1(embeddings))
        x_out = x_out + embeddings
        z = gelu(self.fc2(x_out))
        return self.final(z + x_out)


class SmiTedModel(Module):
    """SMI-TED encode graph: SMILES → embedding, optionally → property."""

    def __init__(self, config: SmiTedModelConfig) -> None:
        self.config = config
        self.encoder = MoLEncoder(config)
        self.decoder = MoLDecoder(config)
        self.net = Net(config)

    def __call__(
        self, input_ids: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        hf = self.config.huggingface_config
        token_embeddings = self.encoder(input_ids, attention_mask)

        mask_expanded = ops.broadcast_to(
            ops.unsqueeze(attention_mask, -1),
            (token_embeddings.shape[0], token_embeddings.shape[1], hf.n_embd),
        )
        masked = token_embeddings * mask_expanded
        flat = ops.reshape(masked, (masked.shape[0], hf.max_len * hf.n_embd))
        embedding = self.decoder.autoencoder.encoder(flat)
        if hf.is_property_output:
            return self.net(embedding)
        return embedding


def build_graph(
    config: SmiTedModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    max_len = config.huggingface_config.max_len
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", max_len], device=config.device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", max_len], device=config.device
    )

    # Embedding serve still constructs ``Net`` so ``net.*`` keys from Hub
    # safetensors load cleanly; they are unused when not in property mode.
    with Graph(
        "smi_ted", input_types=[input_ids_type, attention_mask_type]
    ) as graph:
        model = SmiTedModel(config)
        model.load_state_dict(state_dict)
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        graph.output(model(input_ids, attention_mask))

    return graph
