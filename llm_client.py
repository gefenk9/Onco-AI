import os
import json
import boto3
from openai import AzureOpenAI
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai").lower()  # "azure_openai", "bedrock", or "anthropic"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "o4-mini")

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "eu.anthropic.claude-3-7-sonnet-20250219-v1:0")

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

_azure_openai_client = None
_bedrock_client = None
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
                api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version=AZURE_OPENAI_API_VERSION
            )
            print("Azure OpenAI client initialized.")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            raise
    return _azure_openai_client


def get_bedrock_client(region_name=None):
    """Initializes and returns a Bedrock runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        try:
            client_region = region_name or AWS_REGION
            _bedrock_client = boto3.client('bedrock-runtime', region_name=client_region)
            print(f"Bedrock client initialized for region: {client_region}")
        except Exception as e:
            print(f"Error initializing AWS Bedrock client: {e}")
            raise
    return _bedrock_client


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
    Invokes the configured LLM (Azure OpenAI, Bedrock, or direct Anthropic) and returns the text response.
    Returns the LLM's text response or an error message string starting with "ERROR:".
    """
    current_provider = (provider_override or LLM_PROVIDER).lower()

    try:
        if current_provider == "azure_openai":
            client = get_azure_openai_client()

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_text}]  # type: ignore

            response = client.chat.completions.create(
                model=OPEN_AI_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature  # type: ignore
            )

            if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
                return response.choices[0].message.content
            return f"ERROR: Could not extract LLM response content from Azure OpenAI. Response: {response}"

        elif current_provider == "bedrock":
            client = get_bedrock_client()
            messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}]
            request_body = {
                "system": system_prompt,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            response = client.invoke_model(
                modelId=BEDROCK_MODEL,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(request_body),
            )
            response_body_text = response['body'].read().decode('utf-8')
            response_body_json = json.loads(response_body_text)
            if (
                response_body_json.get("content")
                and len(response_body_json["content"]) > 0
                and "text" in response_body_json["content"][0]
            ):
                return response_body_json["content"][0]['text']
            return f"ERROR: Could not extract LLM response content from Bedrock. Response: {response_body_json}"

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
            return f"ERROR: Unknown LLM provider '{current_provider}'. Choose 'azure_openai', 'bedrock', or 'anthropic' in LLM_PROVIDER env var."
    except Exception as e:
        return f"ERROR: LLM invocation failed for provider '{current_provider}'. Exception: {e}"
