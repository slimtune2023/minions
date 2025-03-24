from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient
from minions.clients.perplexity import PerplexityAIClient
from minions.clients.openrouter import OpenRouterClient
from minions.clients.groq import GroqClient
from minions.clients.deepseek import DeepSeekClient

__all__ = [
    "OllamaClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "AnthropicClient",
    "TogetherClient",
    "PerplexityAIClient",
    "OpenRouterClient",
    "GroqClient",
    "DeepSeekClient",
]

try:
    from minions.clients.mlx_lm import MLXLMClient

    __all__.append("MLXLMClient")
except ImportError:
    # print warning that mlx_lm is not installed
    print(
        "Warning: mlx_lm is not installed. If you want to use mlx_lm, please install it with `pip install mlx-lm`."
    )

try:
    from .cartesia_mlx import CartesiaMLXClient

    __all__.append("CartesiaMLXClient")
except ImportError:
    # If cartesia_mlx is not installed, skip it
    print(
        "Warning: cartesia_mlx is not installed. If you want to use cartesia_mlx, please follow the instructions in the README to install it."
    )


try:
    from .mlx_omni import MLXOmniClient

    __all__.append("MLXOmniClient")
except ImportError:
    # print warning that mlx_omni is not installed
    print(
        "Warning: mlx_omni is not installed. If you want to use mlx_omni, please install it with `pip install mlx-omni-server`"
    )

try:
    from .huggingface import HuggingFaceClient

    __all__.append("HuggingFaceClient")
except ImportError:
    # print warning that huggingface is not installed
    print(
        "Warning: huggingface inference client is not installed. If you want to use huggingface inference client, please install it with `pip install huggingface-hub`"
    )

try:
    from .mlx_audio import MLXAudioClient

    __all__.append("MLXAudioClient")
except ImportError:
    # print warning that mlx_audio is not installed
    print(
        "Warning: mlx_audio is not installed. If you want to use mlx_audio, please install it with `pip install mlx-audio`"
    )
