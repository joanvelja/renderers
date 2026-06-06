from __future__ import annotations

from copy import deepcopy
from typing import Any

_ROPE_FLOAT_FIELDS = frozenset(
    {
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "factor",
        "high_freq_factor",
        "low_freq_factor",
        "mscale",
        "mscale_all_dim",
        "partial_rotary_factor",
        "rope_theta",
    }
)

DIRECT_FAST_TOKENIZER_MODELS = frozenset(
    {
        "Qwen/Qwen3-4B-Instruct-2507",
        "allenai/Olmo-3-7B-Instruct-DPO",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-20b-BF16",
        "google/gemma-4-E2B",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B",
        "google/gemma-4-E4B-it",
        "google/gemma-4-31B",
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B",
        "google/gemma-4-26B-A4B-it",
    }
)


def normalize_legacy_rope_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a Transformers-5-ready config dict without mutating input."""
    normalized = deepcopy(config_dict)
    for key in ("rope_parameters", "rope_scaling"):
        _normalize_rope_params(normalized.get(key))
    return normalized


def _normalize_rope_params(value: Any) -> None:
    if not isinstance(value, dict):
        return

    for nested in value.values():
        if isinstance(nested, dict):
            _normalize_rope_params(nested)

    for key in _ROPE_FLOAT_FIELDS:
        if type(value.get(key)) is int:
            value[key] = float(value[key])


def load_config(pretrained_model_name_or_path: str, **kwargs):
    """Load a config after normalizing legacy RoPE scalar types.

    Some Hub configs are still in the Transformers 4.x ``rope_scaling`` shape.
    Transformers 5.x standardizes that into ``rope_parameters`` before strict
    validation, so integer-valued float fields such as YaRN ``beta_fast`` warn
    unless we normalize the raw dict first.
    """
    from transformers import AutoConfig
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    auto_kwargs = dict(kwargs)
    kwargs = dict(kwargs)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    code_revision = kwargs.pop("code_revision", None)
    kwargs["_from_auto"] = True
    kwargs["name_or_path"] = pretrained_model_name_or_path

    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
    config_dict = normalize_legacy_rope_config(config_dict)

    has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
    if has_remote_code and trust_remote_code:
        if code_revision is not None:
            auto_kwargs["code_revision"] = code_revision
        auto_kwargs["trust_remote_code"] = trust_remote_code
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **auto_kwargs)

    model_type = config_dict.get("model_type")
    if model_type in CONFIG_MAPPING:
        return CONFIG_MAPPING[model_type].from_dict(config_dict, **unused_kwargs)

    if trust_remote_code is not None:
        auto_kwargs["trust_remote_code"] = trust_remote_code
    if code_revision is not None:
        auto_kwargs["code_revision"] = code_revision
    return AutoConfig.from_pretrained(pretrained_model_name_or_path, **auto_kwargs)


def should_use_direct_fast_tokenizer(pretrained_model_name_or_path: str) -> bool:
    return pretrained_model_name_or_path in DIRECT_FAST_TOKENIZER_MODELS


def load_direct_fast_tokenizer(pretrained_model_name_or_path: str, **kwargs):
    """Load a tokenizer JSON directly, bypassing AutoTokenizer dispatch.

    Transformers 5 ``AutoTokenizer`` imports the optional SentencePiece backend
    during dispatch when sentencepiece is installed, even for models that load
    from ``tokenizer.json``. For OLMo3 this produces unrelated SWIG deprecation
    warnings in clean warning runs. The direct fast-tokenizer backend is the
    standard path for tokenizer-JSON models and avoids importing that optional
    slow-tokenizer backend.
    """
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
