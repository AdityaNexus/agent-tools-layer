from dataclasses import dataclass

from .providers import assistant_message, get_text, parse_tool_calls, results_to_messages
from .types import ToolResult


@dataclass
class AgentRun:
    answer: str
    status: str
    messages: list
    tool_results: list[ToolResult]
    steps: int


class ToolAgent:
    def __init__(self, llm, registry, provider="openai", max_steps=8, max_consecutive_errors=3):
        self.llm = llm
        self.registry = registry
        self.provider = provider
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors

    def run(self, prompt):
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        )
        results: list[ToolResult] = []
        error_in_a_row = 0

        for steps in range(1, self.max_steps + 1):
            response = self.llm(messages, self.registry.schemas())
            calls = parse_tool_calls(response, self.provider)
            messages.append(assistant_message(response, self.provider))
            if not calls:
                return AgentRun(
                    answer=get_text(response, self.provider),
                    status="completed",
                    messages=messages,
                    tool_results=results,
                    steps=steps,
                )

            step_results = self.registry.execute_all(calls)
            results.extend(step_results)
            messages.extend(results_to_messages(step_results, self.provider))

            error_in_a_row = 0 if any(r.ok for r in step_results) else error_in_a_row + 1
            if error_in_a_row >= self.max_consecutive_errors:
                return AgentRun(
                    answer="Too many consecutive tool errors",
                    status="too_many_errors",
                    messages=messages,
                    tool_results=results,
                    steps=steps,
                )
        return AgentRun(
            answer="Max steps reached",
            status="max_steps",
            messages=messages,
            tool_results=results,
            steps=self.max_steps,
        )
