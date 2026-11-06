import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError
from providers import structured_query


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int


class ProviderChecks(unittest.TestCase):
    def test_nvidia_avoids_guided_json_and_validates(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"count": 5}')
        result = structured_query(model, Plan, [HumanMessage(content="count /no_think")])
        self.assertEqual(result.count, 5)
        model.bind.assert_not_called()
        model.with_structured_output.assert_not_called()
        messages = model.invoke.call_args.args[0]
        self.assertNotIn("/no_think", messages[-1].content)
        self.assertIn('"count"', messages[0].content)

    def test_invalid_output_is_not_executed(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"wrong_field": 5}')
        with self.assertRaises(ValidationError):
            structured_query(model, Plan, [])

    def test_fenced_json_is_supported(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='```json\n{"count": 5}\n```')
        self.assertEqual(structured_query(model, Plan, []).count, 5)

    def test_closed_thinking_block_is_not_parsed_as_answer(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='<think>internal text</think>\n{"count":5}')
        self.assertEqual(structured_query(model, Plan, []).count, 5)

    def test_invalid_plan_gets_one_repair_attempt(self):
        model = Mock()
        model.invoke.side_effect = [SimpleNamespace(content='not json'),
                                    SimpleNamespace(content='{"count":5}')]
        self.assertEqual(structured_query(model, Plan, []).count, 5)
        self.assertEqual(model.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
