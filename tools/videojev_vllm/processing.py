"""Static encoder sizing for inputs already prepared by VideoJev."""

from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder


class StaticVideoInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"video": 1}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        # Match the existing CPU preparation's complete-window video budget.
        return {"video": seq_len * 4 // 5}


class NoSyntheticInputs(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return ""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {}


class PreparedVideoProcessor(BaseMultiModalProcessor):
    # Real requests already contain tokens from the original HF processor.
    requires_tokenizer = False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(self, inputs, timing_ctx):
        raise NotImplementedError("Use the prepared VideoJev EngineCore input path")
