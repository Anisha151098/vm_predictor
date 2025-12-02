"""
Claude Helper Utilities for Google Colab
A collection of helper functions and classes for interacting with Claude API.
"""

from anthropic import Anthropic
from typing import List, Dict, Optional
import json


class ClaudeChat:
    """A chat interface with conversation history management."""

    def __init__(
        self,
        client: Anthropic,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the chat interface.

        Args:
            client: Anthropic client instance
            model: Claude model to use
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt to set behavior
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

    def send_message(self, user_message: str) -> str:
        """
        Send a message and maintain conversation history.

        Args:
            user_message: The message to send to Claude

        Returns:
            Claude's response as a string
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.conversation_history
        }

        # Add system prompt if provided
        if self.system_prompt:
            api_params["system"] = self.system_prompt

        # Get Claude's response
        response = self.client.messages.create(**api_params)

        # Extract response text
        assistant_message = response.content[0].text

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def export_history(self, filepath: str):
        """Export conversation history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

    def load_history(self, filepath: str):
        """Load conversation history from a JSON file."""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)


def chat_with_claude(
    client: Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None
) -> str:
    """
    Send a single message to Claude and get a response.

    Args:
        client: Anthropic client instance
        prompt: The message to send to Claude
        model: The Claude model to use
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt to set behavior

    Returns:
        Claude's response as a string
    """
    api_params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    if system_prompt:
        api_params["system"] = system_prompt

    message = client.messages.create(**api_params)
    return message.content[0].text


def stream_chat(
    client: Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    callback = None
):
    """
    Stream Claude's response in real-time.

    Args:
        client: Anthropic client instance
        prompt: The message to send to Claude
        model: The Claude model to use
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt
        callback: Optional callback function called with each text chunk

    Returns:
        Complete response as a string
    """
    api_params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    if system_prompt:
        api_params["system"] = system_prompt

    full_response = ""

    with client.messages.stream(**api_params) as stream:
        for text in stream.text_stream:
            full_response += text
            if callback:
                callback(text)
            else:
                print(text, end="", flush=True)

    if not callback:
        print()  # New line at the end

    return full_response


class ClaudeModels:
    """Available Claude models with descriptions."""

    SONNET_4_5 = "claude-sonnet-4-5-20250929"
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    HAIKU_3_5 = "claude-3-5-haiku-20241022"

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """Return a dictionary of available models with descriptions."""
        return {
            "Sonnet 4.5": {
                "id": cls.SONNET_4_5,
                "description": "Latest, most capable model for complex tasks"
            },
            "Sonnet 3.5": {
                "id": cls.SONNET_3_5,
                "description": "Previous generation, still very capable"
            },
            "Haiku 3.5": {
                "id": cls.HAIKU_3_5,
                "description": "Faster, more efficient for simpler tasks"
            }
        }


def setup_claude_client(api_key: str) -> Anthropic:
    """
    Initialize and return an Anthropic client.

    Args:
        api_key: Your Anthropic API key

    Returns:
        Configured Anthropic client
    """
    return Anthropic(api_key=api_key)


def get_api_key_from_colab():
    """
    Attempt to retrieve API key from Google Colab secrets.

    Returns:
        API key string or None if not found
    """
    try:
        from google.colab import userdata
        return userdata.get('ANTHROPIC_API_KEY')
    except ImportError:
        print("Not running in Google Colab")
        return None
    except Exception as e:
        print(f"Error retrieving API key from Colab secrets: {e}")
        return None


# Example usage and quick start function
def quick_start():
    """
    Quick start function to set up Claude in Colab.

    Returns:
        Tuple of (client, chat) ready to use
    """
    # Try to get API key from Colab secrets
    api_key = get_api_key_from_colab()

    if not api_key:
        api_key = input("Enter your Anthropic API key: ")

    # Initialize client
    client = setup_claude_client(api_key)

    # Create chat instance
    chat = ClaudeChat(client)

    print("✓ Claude is ready to use!")
    print("Example usage:")
    print("  response = chat.send_message('Hello, Claude!')")
    print("  print(response)")

    return client, chat


if __name__ == "__main__":
    print("Claude Helper Utilities")
    print("-" * 50)
    print("\nAvailable Models:")
    for name, info in ClaudeModels.list_models().items():
        print(f"  {name}: {info['description']}")
        print(f"    Model ID: {info['id']}")
