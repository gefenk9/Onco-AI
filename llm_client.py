import os
from openai import AzureOpenAI
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai").lower()  # "azure_openai" or "anthropic"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "o4-mini")

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

_azure_openai_client = None
_anthropic_client = None


def get_azure_openai_client():
    """Initializes and returns an Azure OpenAI client."""
    global _azure_openai_client
    if _azure_openai_client is None:
        if not AZURE_OPENAI_API_KEY:
            raise ValueError(
                "AZURE_OPENAI_API_KEY not found. Please set it in your .env file or environment variables."
            )
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not found. Please set it in your .env file or environment variables."
            )
        try:
            _azure_openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version="2024-02-01"
            )
            print("Azure OpenAI client initialized.")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            raise
    return _azure_openai_client


def get_anthropic_client():
    """Initializes and returns an Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.")
        try:
            _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print("Anthropic client initialized.")
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            raise
    return _anthropic_client


def invoke_llm(
    system_prompt: str,
    user_prompt_text: str,
    max_tokens: int,
    temperature: float,
    provider_override: str = None,  # Optionally override LLM_PROVIDER from env # type: ignore
):
    """
    Invokes the configured LLM (Azure OpenAI or direct Anthropic) and returns the text response.
    Returns the LLM's text response or an error message string starting with "ERROR:".
    """
    current_provider = (provider_override or LLM_PROVIDER).lower()

    try:
        if current_provider == "azure_openai":
            client = get_azure_openai_client()

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_text}]

            response = client.chat.completions.create(
                model=OPEN_AI_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature
            )

            if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
                return response.choices[0].message.content
            return f"ERROR: Could not extract LLM response content from Azure OpenAI. Response: {response}"

        elif current_provider == "anthropic":
            client = get_anthropic_client()
            messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}]

            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                system=system_prompt,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response.content and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                return response.content[0].text  # type: ignore
            return f"ERROR: Could not extract LLM response content from Anthropic. Response: {response}"
        else:
            return f"ERROR: Unknown LLM provider '{current_provider}'. Choose 'azure_openai' or 'anthropic' in LLM_PROVIDER env var."
    except Exception as e:
        return f"ERROR: LLM invocation failed for provider '{current_provider}'. Exception: {e}"
