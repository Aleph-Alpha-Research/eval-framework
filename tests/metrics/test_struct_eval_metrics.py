from pytest import fixture

from eval_framework.metrics.completion.struct_eval_metrics import (
    RenderableStructMetric,
    RenderableStructMetricContext,
    StructMetric,
    StructMetricContext,
)
from eval_framework.shared.types import Completion


class TestJSONMetric:
    @fixture
    def json_metric(self) -> StructMetric:
        return StructMetric()

    def test_computes_valid_score(self, json_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='{"key": "value"}',
            raw_completion='{"key": "value"}',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="json", paths=["key"]),
        )
        score = json_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_invalid_format_score(self, json_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='{"key": "value"',
            raw_completion='{"key": "value"',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="json", paths=["key"]),
        )
        score = json_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_subtly_invalid_score(self, json_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='{"wrong_key": "value"}',
            raw_completion='{"wrong_key": "value"}',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="json", paths=["key"]),
        )
        score = json_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, json_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='{"key1": "value1", "key3": "value3"}',
            raw_completion='{"key1": "value1", "key3": "value3"}',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="json", paths=["key1", "key2"]),
        )
        score = json_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "StructMetric/has_keywords"


class TestYAMLMetric:
    @fixture
    def yaml_metric(self) -> StructMetric:
        return StructMetric()

    def test_computes_valid_score(self, yaml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="key: value",
            raw_completion="key: value",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="yaml", paths=["key"]),
        )
        score = yaml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_invalid_format_score(self, yaml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="key: [value",
            raw_completion="key: [value",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="yaml", paths=["key"]),
        )
        score = yaml_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_subtly_invalid_score(self, yaml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="wrong_key: value",
            raw_completion="wrong_key: value",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="yaml", paths=["key"]),
        )
        score = yaml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, yaml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="key1: value1\nkey3: value3",
            raw_completion="key1: value1\nkey3: value3",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="yaml", paths=["key1", "key2"]),
        )
        score = yaml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "StructMetric/has_keywords"


class TestTOMLMetric:
    @fixture
    def toml_metric(self) -> StructMetric:
        return StructMetric()

    def test_computes_valid_score(self, toml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='key = "value"',
            raw_completion='key = "value"',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="toml", paths=["key"]),
        )
        score = toml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_invalid_format_score(self, toml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="key = value",
            raw_completion="key = value",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="toml", paths=["key"]),
        )
        score = toml_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_subtly_invalid_score(self, toml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='wrong_key = "value"',
            raw_completion='wrong_key = "value"',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="toml", paths=["key"]),
        )
        score = toml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, toml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='key1 = "value1"\nkey3 = "value3"',
            raw_completion='key1 = "value1"\nkey3 = "value3"',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="toml", paths=["key1", "key2"]),
        )
        score = toml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "StructMetric/has_keywords"


class TestXMLMetric:
    @fixture
    def xml_metric(self) -> StructMetric:
        return StructMetric()

    def test_computes_valid_score(self, xml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<key>value</key>",
            raw_completion="<key>value</key>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="xml", paths=["key"]),
        )
        score = xml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_invalid_format_score(self, xml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<key>value</key",
            raw_completion="<key>value</key",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="xml", paths=["key"]),
        )
        score = xml_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_subtly_invalid_score(self, xml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<wrong_key>value</wrong_key>",
            raw_completion="<wrong_key>value</wrong_key>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="xml", paths=["key"]),
        )
        score = xml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, xml_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<root><key1>value1</key1><key3>value3</key3></root>",
            raw_completion="<root><key1>value1</key1><key3>value3</key3></root>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="xml", paths=["root.key1", "root.key2"]),
        )
        score = xml_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "StructMetric/has_keywords"


class TestCSVMetric:
    @fixture
    def csv_metric(self) -> StructMetric:
        return StructMetric()

    def test_computes_valid_score(self, csv_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='"head1","head2"\n"val1","val2"\n',
            raw_completion='"head1","head2"\n"val1","val2"\n',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="csv", paths=["csv::head1"]),
        )
        score = csv_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_invalid_format_score(self, csv_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='"unclosed quote',
            raw_completion='"unclosed quote',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="csv", paths=["csv::head1"]),
        )
        score = csv_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_subtly_invalid_score(self, csv_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='"wrong_head1","head2"\n"val1","val2"\n',
            raw_completion='"wrong_head1","head2"\n"val1","val2"\n',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="csv", paths=["csv::head1"]),
        )
        score = csv_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, csv_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='"head1","head3"\n"val1","val2"\n',
            raw_completion='"head1","head3"\n"val1","val2"\n',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="csv", paths=["csv::head1", "csv::head2"]),
        )
        score = csv_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "StructMetric/has_keywords"

    def test_handles_empty_paths(self, csv_metric: StructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion='"head1","head2"\n"val1","val2"\n',
            raw_completion='"head1","head2"\n"val1","val2"\n',
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=StructMetricContext(output_type="csv", paths=[]),
        )
        score = csv_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "StructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "StructMetric/has_keywords"


class TestHTMLMetric:
    @fixture
    def renderable_metric(self) -> RenderableStructMetric:
        return RenderableStructMetric()

    def test_computes_valid_score(self, renderable_metric: RenderableStructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<html><body><h1>Hello</h1><p>World</p></body></html>",
            raw_completion="<html><body><h1>Hello</h1><p>World</p></body></html>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=RenderableStructMetricContext(output_type="html", keywords=["hello", "world"]),
        )
        score = renderable_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "RenderableStructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "RenderableStructMetric/has_keywords"

    def test_computes_partial_keywords_score(self, renderable_metric: RenderableStructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<html><body><h1>Hello</h1></body></html>",
            raw_completion="<html><body><h1>Hello</h1></body></html>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=RenderableStructMetricContext(output_type="html", keywords=["hello", "world"]),
        )
        score = renderable_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "RenderableStructMetric/valid_format"
        assert score[1].value == 0.5
        assert score[1].metric_name == "RenderableStructMetric/has_keywords"

    def test_computes_invalid_score(self, renderable_metric: RenderableStructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<p><div>Broken!<as/p></daslkdjfiv>",
            raw_completion="<p><div>Broken!<as/p></daslkdjfiv>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=RenderableStructMetricContext(output_type="html", keywords=["hello"]),
        )
        score = renderable_metric.calculate(completion)
        assert score[0].value == 0.0
        assert score[0].metric_name == "RenderableStructMetric/valid_format"
        assert score[1].value == 0.0
        assert score[1].metric_name == "RenderableStructMetric/has_keywords"

    def test_handles_empty_keywords(self, renderable_metric: RenderableStructMetric) -> None:
        completion = Completion(
            prompt="",
            messages=None,
            completion="<html><body><h1>Hello</h1></body></html>",
            raw_completion="<html><body><h1>Hello</h1></body></html>",
            prompt_sequence_positions=None,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth="",
            context=RenderableStructMetricContext(output_type="html", keywords=[]),
        )
        score = renderable_metric.calculate(completion)
        assert score[0].value == 1.0
        assert score[0].metric_name == "RenderableStructMetric/valid_format"
        assert score[1].value == 1.0
        assert score[1].metric_name == "RenderableStructMetric/has_keywords"
