try:
    from renderers._version import __version__
except ImportError:
    # Source checkout without a built artifact (e.g. editable install
    # before the first ``uv build`` populates ``_version.py``). Real
    # installs always have it.
    __version__ = "0+unknown"

from renderers.base import (
    MULTIMODAL_MODELS,
    Content,
    ContentPart,
    ImagePart,
    Message,
    MultiModalData,
    MultimodalRenderer,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    RenderedConversation,
    RenderedTokens,
    Renderer,
    RendererPool,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolCallFunction,
    ToolCallParseStatus,
    ToolSpec,
    VideoPart,
    build_training_sample,
    build_trajectory_step,
    create_renderer,
    create_renderer_pool,
    is_multimodal,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.deepseek_v3 import DeepSeekV3Renderer
from renderers.default import DefaultRenderer
from renderers.glm5 import GLM5Renderer
from renderers.glm45 import GLM45Renderer
from renderers.gpt_oss import GptOssRenderer
from renderers.kimi_k2 import KimiK2Renderer
from renderers.kimi_k25 import KimiK25Renderer
from renderers.laguna_xs2 import LagunaXS2Renderer
from renderers.minimax_m2 import MiniMaxM2Renderer
from renderers.nemotron3 import Nemotron3Renderer
from renderers.qwen3 import Qwen3Renderer
from renderers.qwen3_vl import Qwen3VLRenderer
from renderers.qwen35 import Qwen35Renderer
from renderers.qwen36 import Qwen36Renderer
from renderers.streams import (
    CompletedResponse,
    PreparedTurn,
    RenderedStream,
    StreamBridgeUnavailable,
    StreamDivergence,
    StreamSet,
)

__all__ = [
    "Content",
    "ContentPart",
    "CompletedResponse",
    "DeepSeekV3Renderer",
    "DefaultRenderer",
    "GLM45Renderer",
    "GLM5Renderer",
    "GptOssRenderer",
    "ImagePart",
    "KimiK2Renderer",
    "KimiK25Renderer",
    "LagunaXS2Renderer",
    "MULTIMODAL_MODELS",
    "Message",
    "MiniMaxM2Renderer",
    "MultiModalData",
    "MultimodalRenderer",
    "Nemotron3Renderer",
    "ParsedResponse",
    "ParsedToolCall",
    "PlaceholderRange",
    "PreparedTurn",
    "Qwen3Renderer",
    "Qwen3VLRenderer",
    "Qwen35Renderer",
    "Qwen36Renderer",
    "RenderedStream",
    "RenderedConversation",
    "RenderedTokens",
    "Renderer",
    "RendererPool",
    "StreamBridgeUnavailable",
    "StreamDivergence",
    "StreamSet",
    "TextPart",
    "ThinkingPart",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallParseStatus",
    "ToolSpec",
    "VideoPart",
    "__version__",
    "build_training_sample",
    "build_trajectory_step",
    "create_renderer",
    "create_renderer_pool",
    "is_multimodal",
    "reject_assistant_in_extension",
    "trim_to_turn_close",
]
