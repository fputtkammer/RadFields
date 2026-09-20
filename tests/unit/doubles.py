from pydantic import BaseModel
from typing import Type
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage

class StubLLM:
    """
    Test double class, whose instances provide preset answers.
    """
    def __init__(self) -> None:
        self.answers = dict()
        self.template: Type[BaseModel] | None = None

    def with_structured_output(self, schema: Type[BaseModel], method: str | None = None):
        assert issubclass(schema, BaseModel)
        if method is not None:
            assert method == "json_schema"
        new_llm = StubLLM()
        new_llm.template = schema
        new_llm.answers = self.answers
        return new_llm

    def invoke(self, input):
        category_type = list(self.template.model_fields.keys())[0]
        return self.template(**{category_type: self.answers[category_type]})

    def with_retry(self, retry_if_exception_type: tuple[Type[Exception]]):
        assert isinstance(retry_if_exception_type, tuple)
        for exception_type in retry_if_exception_type:
            assert issubclass(exception_type, Exception)
        return self.invoke


class MockLLM:
    """
    Test double class, whose instances test whether prompt includes expected context for target response template.
    Preset answers can be provided to receive all prompts, that should be tested.
    """
    def __init__(self, template_to_context: dict[str, str]) -> None:
        self.template_to_context = template_to_context
        self.answers: dict[str, str | list[str]] = dict()
        self.template: Type[BaseModel] | None = None

    def with_structured_output(self, schema: Type[BaseModel], method: str | None = None):
        assert issubclass(schema, BaseModel)
        if method is not None:
            assert method == "json_schema"
        new_llm = MockLLM(self.template_to_context)
        new_llm.template = schema
        new_llm.answers = self.answers
        return new_llm

    def invoke(self, input: ChatPromptValue):
        current_template_name = self.template.__name__
        if current_template_name in self.template_to_context:
            expected_context = self.template_to_context[current_template_name]
            if hasattr(input, 'messages'):
                searchable_text = "\n".join(str(msg.content) for msg in input.messages)
            else:
                raise ValueError(f"Expected input of type \"ChatPromptValue\", received input of type \"{type(input)}\" instead: {input}")
            assert expected_context in searchable_text
        if self.answers:
            category_type = list(self.template.model_fields.keys())[0]
            return self.template(**{category_type: self.answers[category_type]})
        else:
            return self.template()

    def with_retry(self, retry_if_exception_type: tuple[Type[Exception]]):
        assert isinstance(retry_if_exception_type, tuple)
        for exception_type in retry_if_exception_type:
            assert issubclass(exception_type, Exception)
        return self.invoke


class SequentialMockLLM:
    """
    test double class for structure_report_sequentially: records messages and responses at each invocation.
    Unlike MockLLM, with_retry returns self (not self.invoke) to support direct llm.invoke(messages) calls.
    """
    def __init__(self) -> None:
        self.recorded_messages: list[list[BaseMessage]] = []
        self.recorded_responses: list[str] = []
        self.recorded_schemas: list[Type[BaseModel]] = []
        self.template: Type[BaseModel] | None = None

    def with_structured_output(self, schema: Type[BaseModel], method: str | None = None):
        assert issubclass(schema, BaseModel)
        new_llm = SequentialMockLLM()
        new_llm.template = schema
        new_llm.recorded_messages = self.recorded_messages
        new_llm.recorded_responses = self.recorded_responses
        new_llm.recorded_schemas = self.recorded_schemas
        return new_llm

    def with_retry(self, retry_if_exception_type: tuple[Type[Exception]]):
        assert isinstance(retry_if_exception_type, tuple)
        for exception_type in retry_if_exception_type:
            assert issubclass(exception_type, Exception)
        return self

    def invoke(self, messages: list[BaseMessage]):
        self.recorded_messages.append(list(messages))
        self.recorded_schemas.append(self.template)
        response = self.template()
        self.recorded_responses.append(response.model_dump_json())
        return response
