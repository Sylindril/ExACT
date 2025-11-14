"""
QWEN-vLLM Wrapper for Visual Web Arena
This wrapper provides an interface between ExACT agents and a vLLM-hosted QWEN model.
"""
from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional, Union
import json
import logging
import time

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging to suppress debug messages for this module
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.WARNING)

TIMEOUT = 1000


class Qwen_VLLM():
    """
    Wraps a Qwen2-VL model hosted via vLLM for ExACT agents.
    Provides a clean interface for vision-language inference.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:9001/v1",
        api_key: str = "qwen",
        model_name: str = "qwen_vllm",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        timeout: int = TIMEOUT,
        **kwargs,
    ):
        """
        Initialize the QWEN-vLLM wrapper.

        Args:
            api_base: Base URL for the vLLM server
            api_key: API key for authentication
            model_name: Model name as configured in vLLM
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Repetition penalty
            timeout: Request timeout in seconds
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.timeout = timeout

        # Initialize OpenAI client for vLLM
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

        # Wandb logging
        self.use_wandb = WANDB_AVAILABLE
        self.generation_count = 0

        logging.info(f"Initialized QWEN-vLLM wrapper with base_url={self.api_base}, model={self.model_name}")

    def _encode_image(self, img: Image.Image) -> str:
        """
        Encode a PIL Image to base64 string for API transmission.

        Args:
            img: PIL Image object

        Returns:
            Base64-encoded image string with data URI prefix
        """
        with BytesIO() as image_buffer:
            img.save(image_buffer, format="PNG")
            byte_data = image_buffer.getvalue()
            img_b64 = base64.b64encode(byte_data).decode("utf-8")
            img_b64 = "data:image/png;base64," + img_b64
        return img_b64

    def generate(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a response from the model given a list of messages.

        Args:
            messages: List of message dicts in OpenAI format
                     [{"role": "system/user/assistant", "content": "..."}]
            temperature: Override default temperature
            max_tokens: Override default max tokens
            top_p: Override default top_p

        Returns:
            Generated text response
        """
        # Use instance defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_new_tokens
        top_p = top_p if top_p is not None else self.top_p

        start_time = time.time()

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=self.frequency_penalty,
            )

            answer = completion.choices[0].message.content
            generation_time = time.time() - start_time

            # Log to wandb if available
            if self.use_wandb:
                self._log_to_wandb(
                    generation_time=generation_time,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    completion=completion,
                    output_length=len(answer),
                    num_messages=len(messages),
                )

            return answer

        except Exception as e:
            logging.error(f"Error during generation: {e}")
            raise

    def generate_from_image_and_text(
        self,
        image: Image.Image,
        text: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from an image and text prompt.

        Args:
            image: PIL Image
            text: Text prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add user message with image and text
        base64_image = self._encode_image(image)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image}
                },
                {
                    "type": "text",
                    "text": text
                }
            ]
        })

        return self.generate(messages, **kwargs)

    def generate_from_conversation(
        self,
        conversation_history: List[Tuple[str, Optional[Image.Image]]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from a conversation history.

        Args:
            conversation_history: List of (text, optional_image) tuples
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Process conversation history
        for text, maybe_image in conversation_history:
            if maybe_image is not None:
                # Message with image
                base64_image = self._encode_image(maybe_image)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image}
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                })
            else:
                # Text-only message
                messages.append({
                    "role": "user",
                    "content": text
                })

        return self.generate(messages, **kwargs)

    def _log_to_wandb(
        self,
        generation_time: float,
        temperature: float,
        max_tokens: int,
        top_p: float,
        completion,
        output_length: int,
        num_messages: int,
    ):
        """
        Log generation metrics to Weights & Biases.

        Args:
            generation_time: Time taken for generation
            temperature: Sampling temperature used
            max_tokens: Max tokens parameter
            top_p: Top-p parameter
            completion: API completion object
            output_length: Length of generated output
            num_messages: Number of messages in conversation
        """
        try:
            self.generation_count += 1

            generation_stats = {
                "vlm_generation_time": generation_time,
                "vlm_temperature": temperature,
                "vlm_max_tokens": max_tokens,
                "vlm_top_p": top_p,
                "vlm_frequency_penalty": self.frequency_penalty,
                "vlm_output_length": output_length,
                "vlm_num_messages": num_messages,
                "vlm_generation_count": self.generation_count,
            }

            # Token usage if available in response
            if hasattr(completion, 'usage') and completion.usage:
                generation_stats.update({
                    "vlm_prompt_tokens": completion.usage.prompt_tokens,
                    "vlm_completion_tokens": completion.usage.completion_tokens,
                    "vlm_total_tokens": completion.usage.total_tokens,
                })

            wandb.log(generation_stats)

        except ImportError:
            # wandb not available, silently continue
            pass
        except Exception as e:
            logging.debug(f"VLM wandb logging failed (non-critical): {e}")
            # Disable wandb for future calls to avoid repeated failures
            self.use_wandb = False

    def __repr__(self):
        return f"Qwen_VLLM(api_base={self.api_base}, model={self.model_name})"


# For backward compatibility with the original interface
def create_qwen_vllm_client(
    api_base: str = "http://localhost:9001/v1",
    api_key: str = "qwen",
    **kwargs
) -> Qwen_VLLM:
    """
    Factory function to create a QWEN-vLLM client.

    Args:
        api_base: Base URL for vLLM server
        api_key: API key for authentication
        **kwargs: Additional parameters for Qwen_VLLM

    Returns:
        Initialized Qwen_VLLM instance
    """
    return Qwen_VLLM(api_base=api_base, api_key=api_key, **kwargs)
