"""Tests for the GraphForge A2A protocol module."""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from typing import Any, Dict, Optional

import pytest

from graphforge.a2a._models import (
    A2AAuthenticationError,
    A2AConnectionError,
    A2AError,
    A2AProtocolError,
    A2ATaskError,
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentIcon,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileRef,
    Message,
    Part,
    PushNotification,
    Task,
    TaskCancelRequest,
    TaskCancelResponse,
    TaskGetResponse,
    TaskSendRequest,
    TaskSendResponse,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from graphforge.a2a import (
    A2AClient,
    SyncA2AClient,
    create_a2a_agent_node,
    create_async_a2a_agent_node,
    create_streaming_a2a_agent_node,
)


class TestTextPart:
    def test_create(self) -> None:
        p = TextPart(text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_serialize(self) -> None:
        p = TextPart(text="hello")
        d = p.model_dump(mode="json")
        assert d == {"type": "text", "text": "hello"}

    def test_deserialize(self) -> None:
        p = TextPart.model_validate({"type": "text", "text": "world"})
        assert p.text == "world"


class TestDataPart:
    def test_create(self) -> None:
        p = DataPart(data={"key": "value", "num": 42})
        assert p.type == "data"
        assert p.data["key"] == "value"
        assert p.data["num"] == 42

    def test_serialize(self) -> None:
        p = DataPart(data={"x": 1})
        d = p.model_dump(mode="json")
        assert d == {"type": "data", "data": {"x": 1}}


class TestFilePart:
    def test_create(self) -> None:
        f = FileRef(url="https://example.com/file.png", mime_type="image/png")
        p = FilePart(file=f)
        assert p.type == "file"
        assert p.file.url == "https://example.com/file.png"

    def test_serialize(self) -> None:
        f = FileRef(name="test.png", mimeType="image/png")
        p = FilePart(file=f)
        d = p.model_dump(mode="json", by_alias=True)
        assert d["file"]["mimeType"] == "image/png"
        assert d["file"]["name"] == "test.png"


class TestPart:
    def test_text_part_is_part(self) -> None:
        p: Part = TextPart(text="hi")
        assert isinstance(p, TextPart)

    def test_data_part_is_part(self) -> None:
        p: Part = DataPart(data={"a": 1})
        assert isinstance(p, DataPart)

    def test_file_part_is_part(self) -> None:
        p: Part = FilePart(file=FileRef(url="http://x.com/f"))
        assert isinstance(p, FilePart)


class TestMessage:
    def test_create(self) -> None:
        msg = Message(role="user", parts=[TextPart(text="hello"), DataPart(data={"x": 1})])
        assert msg.role == "user"
        assert len(msg.parts) == 2

    def test_serialize(self) -> None:
        msg = Message(role="agent", parts=[TextPart(text="ok")])
        d = msg.model_dump(mode="json")
        assert d["role"] == "agent"
        assert d["parts"][0]["text"] == "ok"


class TestArtifact:
    def test_create(self) -> None:
        a = Artifact(name="result", parts=[TextPart(text="done")], index=0, last_chunk=False)
        assert a.name == "result"
        assert a.last_chunk is False

    def test_alias(self) -> None:
        a = Artifact.model_validate({"name": "r", "lastChunk": False, "index": 1})
        assert a.last_chunk is False
        assert a.index == 1


class TestTask:
    def test_default_id_generated(self) -> None:
        t = Task()
        assert t.id is not None
        assert len(t.id) > 0

    def test_status_default(self) -> None:
        t = Task()
        assert t.status == TaskStatus.SUBMITTED

    def test_lifecycle(self) -> None:
        t = Task(status=TaskStatus.WORKING)
        assert t.status == TaskStatus.WORKING
        t2 = t.model_copy(update={"status": TaskStatus.COMPLETED})
        assert t2.status == TaskStatus.COMPLETED

    def test_all_status_values(self) -> None:
        for s in TaskStatus:
            t = Task(status=s)
            assert t.status == s

    def test_serialize(self) -> None:
        t = Task(id="test-1", status=TaskStatus.COMPLETED)
        d = t.model_dump(mode="json", by_alias=True)
        assert d["id"] == "test-1"
        assert d["status"] == "completed"
        assert "statusChangedAt" in d


class TestAgentCard:
    def test_minimal(self) -> None:
        c = AgentCard(name="TestBot")
        assert c.name == "TestBot"
        assert c.version == "1.0.0"
        assert c.capabilities.streaming is True

    def test_full(self) -> None:
        c = AgentCard(
            name="FullBot",
            description="A full agent",
            url="http://localhost:8080",
            provider=AgentProvider(organization="Acme"),
            version="2.0.0",
            capabilities=AgentCapabilities(
                skills=[AgentSkill(id="skill-1", name="Greeting")],
                streaming=True,
                push_notifications=True,
            ),
            authentication=AgentAuthentication(schemes=["bearer"]),
            default_input_modes=["text", "data"],
            default_output_modes=["text"],
            icons=[AgentIcon(url="http://x.com/icon.png", width=32, height=32)],
        )
        assert c.name == "FullBot"
        assert c.provider is not None
        assert c.provider.organization == "Acme"
        assert len(c.capabilities.skills) == 1
        assert c.capabilities.skills[0].id == "skill-1"
        assert c.authentication is not None
        assert "bearer" in c.authentication.schemes
        assert len(c.icons) == 1

    def test_serialize_deserialize(self) -> None:
        c1 = AgentCard(name="RoundTrip", capabilities=AgentCapabilities(skills=[AgentSkill(id="s1", name="Skill 1")]))
        d = c1.model_dump(mode="json", by_alias=True)
        c2 = AgentCard.model_validate(d)
        assert c2.name == c1.name
        assert c2.capabilities.skills[0].id == "s1"

    def test_alias_fields(self) -> None:
        c = AgentCard.model_validate({
            "name": "AliasBot",
            "defaultInputModes": ["text", "data"],
            "defaultOutputModes": ["text"],
        })
        assert c.default_input_modes == ["text", "data"]
        assert c.default_output_modes == ["text"]


class TestRequestResponse:
    def test_task_send_request(self) -> None:
        req = TaskSendRequest(id="req-1", message=Message(role="user", parts=[TextPart(text="hi")]), metadata={"source": "test"})
        assert req.id == "req-1"
        assert req.message.role == "user"

    def test_task_send_response(self) -> None:
        t = Task(id="t-1", status=TaskStatus.COMPLETED)
        resp = TaskSendResponse(task=t)
        assert resp.task.status == TaskStatus.COMPLETED

    def test_task_status_update_event(self) -> None:
        ev = TaskStatusUpdateEvent(id="t-1", status=TaskStatus.WORKING, final=False)
        assert ev.final is False
        ev2 = ev.model_copy(update={"status": TaskStatus.COMPLETED, "final": True})
        assert ev2.final is True

    def test_task_get_response(self) -> None:
        t = Task(id="t-1")
        resp = TaskGetResponse(task=t)
        assert resp.task.id == "t-1"

    def test_task_cancel_request(self) -> None:
        req = TaskCancelRequest(metadata={"reason": "test"})
        assert req.metadata["reason"] == "test"

    def test_task_cancel_response(self) -> None:
        resp = TaskCancelResponse(id="t-1", status=TaskStatus.CANCELED)
        assert resp.status == TaskStatus.CANCELED


class TestPushNotification:
    def test_create(self) -> None:
        pn = PushNotification(url="http://callback.example.com", authentication={"token": "abc123"})
        assert pn.url == "http://callback.example.com"
        assert pn.authentication is not None
        assert pn.authentication["token"] == "abc123"


class TestExceptions:
    def test_hierarchy(self) -> None:
        assert issubclass(A2AConnectionError, A2AError)
        assert issubclass(A2AProtocolError, A2AError)
        assert issubclass(A2AAuthenticationError, A2AError)
        assert issubclass(A2ATaskError, A2AError)

    def test_connection_error(self) -> None:
        err = A2AConnectionError("connection refused")
        assert "connection refused" in str(err)

    def test_protocol_error(self) -> None:
        err = A2AProtocolError("bad response")
        assert isinstance(err, A2AError)

    def test_authentication_error(self) -> None:
        err = A2AAuthenticationError("invalid token")
        assert isinstance(err, A2AError)

    def test_task_error(self) -> None:
        err = A2ATaskError("task failed")
        assert isinstance(err, A2AError)
        assert "task failed" in str(err)


class TestJsonRoundTrip:
    def test_message_roundtrip(self) -> None:
        msg = Message(role="user", parts=[TextPart(text="hello"), DataPart(data={"count": 3})], metadata={"session": "abc"})
        d = msg.model_dump(mode="json")
        msg2 = Message.model_validate(d)
        assert msg2.role == msg.role
        assert len(msg2.parts) == 2
        assert isinstance(msg2.parts[0], TextPart)
        assert isinstance(msg2.parts[1], DataPart)

    def test_task_roundtrip(self) -> None:
        t = Task(id="roundtrip-1", status=TaskStatus.WORKING, messages=[Message(role="user", parts=[TextPart(text="hi")])], metadata={"env": "test"})
        d = t.model_dump(mode="json")
        t2 = Task.model_validate(d)
        assert t2.id == t.id
        assert t2.status == TaskStatus.WORKING
        assert len(t2.messages) == 1
        assert t2.metadata["env"] == "test"

    def test_agent_card_roundtrip(self) -> None:
        card = AgentCard(name="TestAgent", version="1.0.0", capabilities=AgentCapabilities(skills=[AgentSkill(id="s1", name="S1")], streaming=True))
        d = card.model_dump(mode="json", by_alias=True)
        card2 = AgentCard.model_validate(d)
        assert card2.name == "TestAgent"
        assert card2.capabilities.streaming is True


class TestAgentNodeFactories:
    def test_create_a2a_agent_node(self) -> None:
        node = create_a2a_agent_node(agent_url="http://localhost:9999")
        assert callable(node)

    def test_create_async_a2a_agent_node(self) -> None:
        node = create_async_a2a_agent_node(agent_url="http://localhost:9999")
        assert callable(node)
        assert asyncio.iscoroutinefunction(node)

    def test_create_streaming_a2a_agent_node(self) -> None:
        node = create_streaming_a2a_agent_node(agent_url="http://localhost:9999")
        assert callable(node)
        assert inspect.isasyncgenfunction(node)

    def test_custom_mappers(self) -> None:
        def input_map(state: Any) -> Message:
            return Message(role="user", parts=[TextPart(text=str(state))])
        def output_map(msg: Any, task: Task) -> Dict[str, Any]:
            return {"result": str(msg)}
        node = create_a2a_agent_node(agent_url="http://localhost:9999", input_mapper=input_map, output_mapper=output_map)
        assert callable(node)

    def test_with_api_key(self) -> None:
        node = create_a2a_agent_node(agent_url="http://localhost:9999", api_key="sk-test")
        assert callable(node)


@pytest.mark.skipif(True, reason="Network binding disabled in sandbox")
class TestIntegration:
    @pytest.mark.asyncio
    async def test_client_server_roundtrip(self) -> None:
        from graphforge import Graph, GraphState, node_field
        from graphforge.a2a._server import A2AServer

        class EchoState(GraphState):
            messages: list = node_field(default=[], merge="append")
            next_step: str = ""

        def echo_node(state: EchoState) -> dict:
            user_msg = "echo: " + str(state.messages[-1]) if state.messages else "empty"
            return {"messages": [{"role": "assistant", "content": user_msg}]}

        graph = (
            Graph[EchoState]()
            .add_node("echo", echo_node)
            .add_edge("echo", "__end__")
            .set_entry_point("echo")
            .compile()
        )
        card = AgentCard(name="EchoAgent", capabilities=AgentCapabilities(skills=[AgentSkill(id="echo", name="Echo")]))
        server = A2AServer(graph, agent_card=card, host="127.0.0.1", port=0)
        await server.start()
        port = server._site._server.sockets[0].getsockname()[1]
        client = A2AClient(f"http://127.0.0.1:{port}")
        try:
            card2 = await client.fetch_agent_card()
            assert card2.name == "EchoAgent"
            task = await client.send_task(Message(role="user", parts=[TextPart(text="hello")]))
            assert task.status == TaskStatus.COMPLETED
        finally:
            await client.close()
            await server.stop()
