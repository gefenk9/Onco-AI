import os
import json
import boto3
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").lower()  # "bedrock" or "anthropic"
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")  # Used if LLM_PROVIDER is "bedrock"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Used if LLM_PROVIDER is "anthropic"

ANTHROPIC_CLAUDE_3_5_SONNET_MODEL_NAME = "claude-3-5-sonnet-20240620"
BEDROCK_CLAUDE_3_5_SONNET_MODEL_NAME = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
# Note: Bedrock model IDs are passed by the calling script

_bedrock_client = None
_anthropic_client = None


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
    Invokes the configured LLM (Bedrock or direct Anthropic) and returns the text response.
    Returns the LLM's text response or an error message string starting with "ERROR:".
    """
    current_provider = (provider_override or LLM_PROVIDER).lower()
    messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}]

    try:
        if current_provider == "bedrock":
            client = get_bedrock_client()
            request_body = {
                "system": system_prompt,
                "anthropic_version": "bedrock-2023-05-31",  # Required for Claude on Bedrock
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            response = client.invoke_model(
                modelId=BEDROCK_CLAUDE_3_5_SONNET_MODEL_NAME,
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
            response = client.messages.create(
                model=ANTHROPIC_CLAUDE_3_5_SONNET_MODEL_NAME,
                system=system_prompt,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response.content and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                return response.content[0].text  # type: ignore
            return f"ERROR: Could not extract LLM response content from Anthropic. Response: {response}"
        else:
            return f"ERROR: Unknown LLM provider '{current_provider}'. Choose 'bedrock' or 'anthropic' in LLM_PROVIDER env var."
    except Exception as e:
        return f"ERROR: LLM invocation failed for provider '{current_provider}'. Exception: {e}"
