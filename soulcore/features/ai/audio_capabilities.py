"""Public audio capability surface.

Provider implementations live in focused modules; callers keep importing this
module so the runtime and external tests retain one stable contract.
"""

from .audio_gsvi_adapter import GSVISpeechCapabilityAdapter
from .audio_provider_adapters import (
    GPTSoVITSSpeechCapabilityAdapter,
    MiMoAudioCapabilityAdapter,
    MiniMaxSpeechCapabilityAdapter,
    OpenAIAudioCapabilityAdapter,
)
from .audio_support import (
    GPTSoVITSSpeechConfig,
    GSVISpeechConfig,
    MiMoAudioConfig,
    MiniMaxSpeechConfig,
    OpenAIAudioConfig,
)
from .audio_transport import AudioHTTPTransport, HTTPAudioResponse, UrllibAudioTransport

__all__ = [
    "AudioHTTPTransport",
    "GPTSoVITSSpeechCapabilityAdapter",
    "GPTSoVITSSpeechConfig",
    "GSVISpeechCapabilityAdapter",
    "GSVISpeechConfig",
    "HTTPAudioResponse",
    "MiMoAudioCapabilityAdapter",
    "MiMoAudioConfig",
    "MiniMaxSpeechCapabilityAdapter",
    "MiniMaxSpeechConfig",
    "OpenAIAudioCapabilityAdapter",
    "OpenAIAudioConfig",
    "UrllibAudioTransport",
]
