import pytest

from agentic_rag.config import Config
from agentic_rag.errors import MissingCredentialsError
from agentic_rag.llm import AnthropicClient, StubClient, get_client


def _config(**overrides) -> Config:
    base = dict(
        anthropic_api_key=None,
        model="claude-opus-5",
        llm_provider="",
        voyage_api_key=None,
        data_dir=None,
        trace_path=None,
        max_steps=6,
        log_level="INFO",
    )
    base.update(overrides)
    return Config(**base)


def test_offline_flag_always_returns_stub_even_with_key() -> None:
    client = get_client(_config(anthropic_api_key="sk-ant-fake"), offline=True)
    assert isinstance(client, StubClient)


def test_no_key_defaults_to_stub() -> None:
    client = get_client(_config())
    assert isinstance(client, StubClient)


def test_key_present_defaults_to_anthropic() -> None:
    client = get_client(_config(anthropic_api_key="sk-ant-fake"))
    assert isinstance(client, AnthropicClient)


def test_explicit_anthropic_without_key_raises() -> None:
    with pytest.raises(MissingCredentialsError):
        get_client(_config(llm_provider="anthropic"))


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_client(_config(llm_provider="not-a-real-provider"))
