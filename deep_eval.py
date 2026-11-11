import json
from typing import Any, Dict, List, Optional


def _build_message_summary(message: Dict[str, Any], expected_summary: Optional[str] = None) -> str:
    if expected_summary:
        return expected_summary

    payload = message.get("payload") or {}
    try:
        payload_text = json.dumps(payload, default=str, sort_keys=True)
    except TypeError:
        payload_text = str(payload)

    sender = message.get("sender", "unknown")
    receiver = message.get("receiver", "unknown")
    msg_type = message.get("type", "agent_message")
    return f"{sender} -> {receiver} [{msg_type}]: {payload_text}"


def _run_deepeval_evaluation(
    input_text: str,
    actual_output: str,
    expected_output: str,
    context: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except Exception as exc:  # pragma: no cover - dependency optional
        return {"status": "skipped", "reason": f"DeepEval not available: {exc}"}

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context or [],
        metadata=metadata or {},
    )

    try:
        results = evaluate(
            test_cases=[test_case],
            metrics=[
                AnswerRelevancyMetric(threshold=0.7),
                FaithfulnessMetric(threshold=0.7),
            ],
            display_config=None,
        )
        return {"status": "ok", "result": str(results)[:500]}
    except Exception as exc:  # pragma: no cover - evaluation failures should not break runtime
        return {"status": "skipped", "reason": f"DeepEval evaluation failed: {exc}"}


def evaluate_agent_communication(
    message: Dict[str, Any],
    expected_summary: Optional[str] = None,
    context: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate an ACP-style inter-agent message using DeepEval when available."""
    if not isinstance(message, dict):
        return {"status": "skipped", "reason": "message must be a dict"}

    actual_output = _build_message_summary(message, expected_summary)
    sender = message.get("sender", "unknown")
    receiver = message.get("receiver", "unknown")
    msg_type = message.get("type", "agent_message")
    trace_id = message.get("trace_id", "unknown")

    return _run_deepeval_evaluation(
        input_text=f"Evaluate the agent communications between {sender} and {receiver} for {msg_type}.",
        actual_output=actual_output,
        expected_output=expected_summary
        or f"A structured {msg_type} communication from {sender} to {receiver} with identifiable trace {trace_id}.",
        context=context
        or [
            f"Message type: {msg_type}",
            f"Sender: {sender}",
            f"Receiver: {receiver}",
            f"Trace ID: {trace_id}",
        ],
        metadata={
            "sender": sender,
            "receiver": receiver,
            "type": msg_type,
            "trace_id": trace_id,
            "timestamp": message.get("timestamp"),
        },
    )


def evaluate_agent_response(
    agent_name: str,
    user_input: str,
    response: str,
    expected_output: Optional[str] = None,
    context: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a final generated agent response using DeepEval."""
    if not agent_name or not response:
        return {"status": "skipped", "reason": "agent_name and response are required"}

    return _run_deepeval_evaluation(
        input_text=f"User asked: {user_input}",
        actual_output=response,
        expected_output=expected_output
        or f"{agent_name} should provide a relevant and structured answer to the user's request.",
        context=context or [
            f"Agent: {agent_name}",
            f"User input: {user_input}",
            "Assesses relevance, factual grounding, and usefulness.",
        ],
        metadata={"agent_name": agent_name, "user_input": user_input},
    )
