"""Original VideoJev vision/readouts around the vLLM Qwen3.5 language executor."""

from types import MethodType, SimpleNamespace

import torch
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model as HFQwenModel,
    Qwen3_5VisionModel,
)
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from tools.videojev_native.contract import MODEL, REVISION
from .processing import NoSyntheticInputs, PreparedVideoProcessor, StaticVideoInfo


def native_positions(config, input_ids, types, grid):
    """Reuse the pinned HF fresh-window integer position calculation."""
    owner = SimpleNamespace(config=config)
    owner.get_vision_position_ids = MethodType(HFQwenModel.get_vision_position_ids, owner)
    positions, delta = HFQwenModel.get_rope_index(
        owner, input_ids, mm_token_type_ids=types, video_grid_thw=grid,
    )
    return positions[:, 0], int(delta[0, 0])


class AttentionOnlyLayer(Qwen3_5DecoderLayer):
    def forward(self, hidden_states, residual, positions, **kwargs):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)
        # The language model's final norm performs this residual addition once.
        return hidden_states, residual


class DecisionPooler(Pooler):
    def __init__(self, head):
        super().__init__()
        # The model owns and loads the module; do not register its weights twice.
        self._score = head.forward

    def get_supported_tasks(self):
        return {"token_embed"}

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        outputs = []
        for index, params in enumerate(pooling_metadata.pooling_params):
            if not cursor.finished_mask[index]:
                outputs.append(None)
                continue
            data = params.extra_kwargs
            device = hidden_states.device
            offset = cursor.first_token_indices_gpu[index]
            pairs = torch.as_tensor(data["pairs"], device=device) + offset
            queries = torch.as_tensor(data["queries"], device=device) + offset
            mask = torch.as_tensor(data["mask"], dtype=torch.bool, device=device)
            tasks = torch.as_tensor(data["tasks"], device=device)
            with torch.autocast("cuda", enabled=False):
                logits = self._score(hidden_states[pairs], hidden_states[queries])
                valid = logits.masked_fill(~mask, -torch.inf)
                probabilities = torch.where(
                    (tasks == 1)[:, None], valid.softmax(-1), logits.sigmoid(),
                ) * mask
            outputs.append(torch.stack((logits, probabilities, mask.float())))
        return outputs


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    PreparedVideoProcessor, info=StaticVideoInfo, dummy_inputs=NoSyntheticInputs,
)
class VideoJevForPooling(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True
    supports_multimodal_pruning = False
    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "backbone.language_model.": "language_model.",
        "backbone.visual.": "visual.",
    })

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = config = vllm_config.model_config.hf_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = self.model_config.multimodal_config
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5Model(
                vllm_config=vllm_config, prefix="language_model",
            )
        last = self.language_model.layers[-1]
        last.__class__ = AttentionOnlyLayer
        del last.mlp
        del last.post_attention_layernorm
        with self._mark_tower_model(vllm_config, {"video"}):
            # Real CPU construction preserves non-persistent rotary buffers.
            # Materializing a meta module would leave those buffers uninitialized.
            with torch.device("cpu"):
                config.vision_config._attn_implementation = "flash_attention_2"
                self.visual = Qwen3_5VisionModel(config.vision_config).float()
            self.visual.to("cuda:0")
        head_class = get_class_from_dynamic_module(
            "modeling_videojev.CosineDecisionHead", MODEL, revision=REVISION,
            cache_dir=vllm_config.load_config.download_dir, local_files_only=True,
        )
        self.head = head_class(
            config.text_config.hidden_size, config.videojev_head_dim,
            config.videojev_head_logit_scale,
        ).float().to("cuda:0")
        self.pooler = DecisionPooler(self.head)
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.visual_events = []
        self.language_events = []
        self.eval().requires_grad_(False)

    @classmethod
    def get_placeholder_str(cls, modality, i):
        return "<|video_pad|>" if modality == "video" else None

    def embed_multimodal(self, **kwargs):
        pixels = kwargs.get("video_pixels")
        if pixels is None:
            return []
        features = []
        for value, grid in zip(pixels, kwargs["video_grid"], strict=True):
            before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            before.record()
            chunks = []
            temporal, height, width = grid.tolist()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, temporal, 16):
                    count = min(16, temporal - start)
                    selected = value[start * height * width:(start + count) * height * width]
                    shape = torch.tensor([[count, height, width]], device="cuda:0")
                    chunks.append(self.visual(
                        selected.to("cuda:0"), grid_thw=shape, return_dict=True,
                    ).pooler_output)
            features.append(torch.cat(chunks))
            after.record()
            self.visual_events.append((before, after))
        return features

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        embeddings = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings):
            embeddings[is_multimodal] = torch.cat(list(multimodal_embeddings)).to(embeddings)
        return embeddings

    def get_mrope_input_positions(self, input_tokens, mm_features):
        data = mm_features[0].data
        return data["native_positions"].data, int(data["native_delta"].data)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        before.record()
        hidden = self.language_model(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds,
        )
        after.record()
        self.language_events.append((before, after))
        return hidden

    def load_weights(self, weights):
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # Native HF loading uses BF16 visual parameters before prepare_for_inference
        # promotes them to FP32. Preserve that rounding, not a new FP32 weight path.
        # Non-persistent rotary buffers retain their original FP32 initialization.
        for parameter in self.visual.parameters():
            parameter.data = parameter.data.bfloat16().float()
        return loaded
