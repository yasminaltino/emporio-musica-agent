import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage


@pytest.fixture
def ask_agent(monkeypatch):
    import langchain.agents
    import langchain_google_genai

    fake_agent = Mock()
    fake_agent.invoke.return_value = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_products", "args": {"name": "Violão"}, "id": "1"}
                ],
            ),
            ToolMessage(
                content="resultado de teste", name="search_products", tool_call_id="1"
            ),
            AIMessage(content="Resposta de teste"),
        ]
    }
    monkeypatch.setattr(langchain.agents, "create_agent", Mock(return_value=fake_agent))
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", Mock())
    module = runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "agent" / "agent.py")
    )
    return module["ask_agent"]


def test_agent_diagnostics_disabled_by_default(ask_agent, capsys):
    assert ask_agent("Pergunta de teste") == "Resposta de teste"
    assert capsys.readouterr().out == ""


def test_agent_diagnostics_enabled(ask_agent, capsys):
    assert ask_agent("Pergunta de teste", debug=True) == "Resposta de teste"
    output = capsys.readouterr().out

    assert "[TOOL]: search_products" in output
    assert "[ARGS]: {'name': 'Violão'}" in output
    assert "[TOOL RESULT]: search_products" in output
    assert "[OUTPUT]: resultado de teste" in output
    assert "[TEMPO TOTAL]:" in output
