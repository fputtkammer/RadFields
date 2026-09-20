import os.path
from copy import deepcopy
import yaml
import pytest

from src.rad_fields.identifier import ReportTypeIdentifier
from tests.unit.doubles import StubLLM, MockLLM

# helper function
def assert_expected_tree(raw: list[str] | dict[str, str | list[str] | dict[str, list[str]]], expected: dict[str, dict[str, list[str]]], tmp_path: str,
                         warning_type: type[Warning] = None) -> None:
    """
    tests whether report type tree determined by ReportTypeIdentifier upon initialization from raw tree / YAML with raw tree as contents is expected tree
    :param raw: raw report type tree
    :param expected: expected processed report type tree
    :param tmp_path: path to save temporary YAML files in
    :param warning_type: type of warning, if warning is expected to be raised or None, otherwise
    """
    file_path = os.path.join(tmp_path, "tmp.yaml")
    yaml.safe_dump(raw, open(file_path, 'w'))
    if warning_type is None:
        # init from raw tree
        ident = ReportTypeIdentifier("dummy", raw_report_type_tree=raw)
        assert ident._report_type_tree == expected
        # init from yaml file
        ident = ReportTypeIdentifier("dummy", initialization_file_path=file_path)
        assert ident._report_type_tree == expected
    else:
        # init from raw tree
        with pytest.warns(warning_type):
            ident = ReportTypeIdentifier("dummy", raw_report_type_tree=raw)
        assert ident._report_type_tree == expected
        # init from yaml file
        with pytest.warns(warning_type):
            ident = ReportTypeIdentifier("dummy", initialization_file_path=file_path)
        assert ident._report_type_tree == expected

# test for __init__
def test_init_raw_input_trees():
    """
    Tests that ReportTypeIdentifier normalises various raw tree formats into the expected internal
    _report_type_tree structure on initialisation, and raises the correct errors for invalid inputs.

    Covers: list of modalities, flat modality→exam_type dict, modality→list-of-exam-types dict,
    fully nested dict with str/list diagnosis values, TypeError for bad top-level or per-modality
    types, FileNotFoundError for missing YAML, and yaml.YAMLError for malformed YAML.
    """
    tmp_path = "tests/data"
    # list of modalities -> warns and yields modalities with empty dicts
    raw_list = ["modA", "modB"]
    expected_from_list = {"modA": dict(), "modB": dict()}
    assert_expected_tree(raw_list, expected_from_list, tmp_path, UserWarning)

    # list of modalities -> warns and yields modalities with single exam dict
    raw_list = {"mod1": "et1", "mod2": "etA"}
    expected_from_list = {"mod1": {"et1": []}, "mod2": {"etA": []}}
    assert_expected_tree(raw_list, expected_from_list, tmp_path)

    # dict of modalities -> list of exam types as value -> warns and produces exam types with empty lists
    raw_dict_list = {"mod1": ["et1", "et2"], "mod2": ["etA"]}
    expected_from_dict_list = {"mod1": {"et1": [], "et2": []}, "mod2": {"etA": []}}
    assert_expected_tree(raw_dict_list, expected_from_dict_list, tmp_path)

    # dict of modalities -> dict of exam types -> exam types map to list of diagnoses (handles str/list/empty)
    raw_complex = {
        "modX": {"et1": ["d1", "d2"], "et2": "d3"},
        "modY": {"etA": "d4"}
    }
    expected_complex = {
        "modX": {"et1": ["d1", "d2"], "et2": ["d3"]},
        "modY": {"etA": ["d4"]}
    }
    assert_expected_tree(raw_complex, expected_complex, tmp_path)
    # test correct exceptions raised in response to invalid inputs
    # top-level raw_report_type_tree of invalid type (not list/dict) -> TypeError
    with pytest.raises(TypeError):
        ReportTypeIdentifier("dummy", raw_report_type_tree=123)

    # modality value of unexpected type (e.g. int) -> TypeError raised during processing
    with pytest.raises(TypeError):
        ReportTypeIdentifier("dummy", raw_report_type_tree={"mod_bad": 999})

    # nonexistent initialization file -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ReportTypeIdentifier("dummy", initialization_file_path="this_file_does_not_exist.yaml")

    # malformed YAML in initialization file -> yaml.YAMLError
    bad_yaml_path = os.path.join(tmp_path, "bad.yaml")
    open(bad_yaml_path, 'w').write("::: not valid yaml :::")
    with pytest.raises(yaml.YAMLError):
        ReportTypeIdentifier("dummy", initialization_file_path=bad_yaml_path)

# tests for identify_report_type method
# -  expected content prepared for requests to inference server
def test_identify_report_type_prompt_content():
    """
    Tests that the prompts sent to the LLM contain both the report text and the response template
    for each classification level (modality, exam type, main diagnosis), for both values of
    categorize_novelties.
    """
    example_report = "CT des Abdomens: Leber unauffällig."
    pipeline_types = ["modality", "exam_type", "main_diagnosis"]
    for categorize_novelties in [True, False]:
        mock_llm = MockLLM({})
        mock_llm.answers = {
            "modalitaet": "modality1",
            "untersuchungstyp": "exam_type1",
            "hauptdiagnose": "diagnosis1",
        }
        identifier = ReportTypeIdentifier(
            "", "tests/data/initial_examples_dummy.yaml",
            categorize_novelties=categorize_novelties, _llm=mock_llm
        )
        # check report text appears in prompt at each classification level
        mock_llm.template_to_context.update({
            identifier._response_templates[pt].__name__: example_report
            for pt in pipeline_types
        })
        identifier.identify_report_type(example_report)
        # check response template appears in prompt at each classification level
        mock_llm.template_to_context.update({
            identifier._response_templates[pt].__name__: identifier._template_read_versions[pt]
            for pt in pipeline_types
        })
        identifier.identify_report_type(example_report)

# - content of responses received from inference server handled appropriately
def test_identify_report_type_existing_categories():
    """
    Tests that when the LLM responds with an existing category at every classification level
    (modality, exam type, main diagnosis), identify_report_type returns the LLM's selection and
    leaves _report_type_tree unchanged, for both categorize_novelties=True and False.
    """
    initial_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    for categorize_novelties in [True, False]:
        spy_llm = StubLLM()
        spy_llm.answers["modalitaet"] = "modality1"
        spy_llm.answers["untersuchungstyp"] = "exam_type1"
        spy_llm.answers["hauptdiagnose"] = "diagnosis1"
        identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                          categorize_novelties=categorize_novelties, _llm=spy_llm)
        identifier._report_type_tree = deepcopy(initial_tree)
        modality, exam_type, diagnosis = identifier.identify_report_type("")
        assert modality == "modality1"
        assert exam_type == "exam_type1"
        assert diagnosis == "diagnosis1"
        assert identifier._report_type_tree == initial_tree


def test_identify_report_type_dynamic_tree_categorize_novelties():
    """
    test whether internal state of ReportTypeIdentifier correctly tracks exam types / diagnoses identified by LLM if categorize_novelties=True
    :return:
    """
    # initialize object where interaction with inference backend is replaced by interaction with deterministic mock object
    mock_llm = StubLLM()
    # case: LLM may add new categories for novelties
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml", _llm=mock_llm)
    # reset report_type_tree
    identifier._report_type_tree = {"modality1": dict(), "modality2": dict(), "modality3": dict()}
    # test newly identified exam types / diagnoses are added to report type tree
    tree_to_add = {"modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1", "diagnosis2"]},
                   "modality2": {"exam_type1": ["diagnosis1"], "exam_type2": ["diagnosis1", "diagnosis2"]},
                   "modality3": {"exam_type1": ["diagnosis1"]}}
    for modality in tree_to_add.keys():
        for exam_type in tree_to_add[modality].keys():
            for diagnosis in tree_to_add[modality][exam_type]:
                mock_llm.answers["modalitaet"] = modality
                mock_llm.answers["untersuchungstyp"] = exam_type
                mock_llm.answers["hauptdiagnose"] = diagnosis
                identifier.identify_report_type("")
    assert identifier._report_type_tree == tree_to_add
    # case: standard open world classification
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml", _llm=mock_llm)
    # reset report_type_tree
    initial_tree = {"modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
                    "modality2": {"exam_type1": ["diagnosis1"]}}
    identifier._report_type_tree = initial_tree
    # test newly identified exam types / diagnoses are added to report type tree
    unique_answers = {"modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1", "diagnosis2"]},
                      "modality2": {"exam_type1": ["diagnosis1"], "exam_type2": ["diagnosis1", "diagnosis2"]}}
    for modality in unique_answers.keys():
        for exam_type in unique_answers[modality].keys():
            for diagnosis in unique_answers[modality][exam_type]:
                mock_llm.answers["modalitaet"] = modality
                mock_llm.answers["untersuchungstyp"] = exam_type
                mock_llm.answers["hauptdiagnose"] = diagnosis
                identifier.identify_report_type("")
    assert identifier._report_type_tree == initial_tree


def test_identify_report_type_static_tree_reject_novelties():
    """
    Tests that with categorize_novelties=False, responses outside preset categories at each level
    (modality, exam type, main diagnosis) are rejected, i.e. return an incomplete response (including default field value(s)) and leave
    the report type tree unchanged.
    """
    mock_llm = StubLLM()
    # initialize with dummy yaml so modality Literal includes modality1/modality2/modality3
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=False, _llm=mock_llm)
    default_value = identifier._special_responses["default_field_value"]
    expected_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    identifier._report_type_tree = deepcopy(expected_tree)
    # case: unknown modality -> other values are never requested, all return values fall back to default, tree unchanged
    # unknown_modality fails Pydantic validation (modality field uses Literal) -> warns
    mock_llm.answers["modalitaet"] = "unknown_modality"
    with pytest.warns(UserWarning, match="LLM failed to respond in accordance with schema"):
        modality, exam_type, diagnosis = identifier.identify_report_type("")
    assert modality == default_value
    assert exam_type == default_value
    assert diagnosis == default_value
    assert identifier._report_type_tree == expected_tree
    # case: unknown exam type -> diagnosis is never requested, exam type and diagnosis both fall back to default, tree unchanged
    # exam_type field uses plain str (no ValidationError); unmatched response triggers its own warning
    mock_llm.answers["modalitaet"] = "modality1"
    mock_llm.answers["untersuchungstyp"] = "unknown_exam_type"
    with pytest.warns(UserWarning, match="LLM failed to provide an appropriate exam type selection response"):
        modality, exam_type, diagnosis = identifier.identify_report_type("")
    assert modality == "modality1"
    assert exam_type == default_value
    assert diagnosis == default_value
    assert identifier._report_type_tree == expected_tree
    # case: unknown diagnosis -> only diagnosis falls back to default, tree unchanged
    # main_diagnosis field uses plain str (no ValidationError); unmatched response triggers its own warning
    mock_llm.answers["modalitaet"] = "modality1"
    mock_llm.answers["untersuchungstyp"] = "exam_type1"
    mock_llm.answers["hauptdiagnose"] = "unknown_diagnosis"
    with pytest.warns(UserWarning, match="LLM failed to provide an appropriate main diagnosis selection response"):
        modality, exam_type, diagnosis = identifier.identify_report_type("")
    assert modality == "modality1"
    assert exam_type == "exam_type1"
    assert diagnosis == default_value
    assert identifier._report_type_tree == expected_tree
    # sanity check: known values at all levels -> complete response, tree unchanged
    mock_llm.answers["modalitaet"] = "modality1"
    mock_llm.answers["untersuchungstyp"] = "exam_type1"
    mock_llm.answers["hauptdiagnose"] = "diagnosis1"
    modality, exam_type, diagnosis = identifier.identify_report_type("")
    assert modality == "modality1"
    assert exam_type == "exam_type1"
    assert diagnosis == "diagnosis1"
    assert identifier._report_type_tree == expected_tree


def test_identify_report_type_sentinel_responses():
    """
    Tests that when the LLM returns a special sentinel value (no_findings_response, other_response,
    multiple_exams_response, default_field_value) at any classification level, identify_report_type:
    - returns the sentinel value at that level and default values for all downstream levels,
    - sends no further requests to the LLM beyond the level where the sentinel was received,
    - leaves the report type tree unchanged.
    """
    spy_llm = StubLLM()
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=False, _llm=spy_llm)
    default_value = identifier._special_responses["default_field_value"]
    no_findings = identifier._special_responses["no_findings_response"]
    other = identifier._special_responses["other_response"]
    multiple_exams = identifier._special_responses["multiple_exams_response"]
    sentinels = [no_findings, other, multiple_exams, default_value]
    expected_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    identifier._report_type_tree = deepcopy(expected_tree)
    # case: sentinel at modality level -> exam type & diagnosis not queried (unset keys would raise KeyError),
    # modality returns sentinel, downstream levels fall back to default, tree unchanged
    for sentinel in sentinels:
        spy_llm.answers["modalitaet"] = sentinel
        modality, exam_type, diagnosis = identifier.identify_report_type("")
        assert modality == sentinel
        assert exam_type == default_value
        assert diagnosis == default_value
        assert identifier._report_type_tree == expected_tree
    # case: sentinel at exam type level -> diagnosis not queried (unset key would raise KeyError),
    # exam type returns sentinel, diagnosis falls back to default, tree unchanged
    for sentinel in sentinels:
        spy_llm.answers["modalitaet"] = "modality1"
        spy_llm.answers["untersuchungstyp"] = sentinel
        modality, exam_type, diagnosis = identifier.identify_report_type("")
        assert modality == "modality1"
        assert exam_type == sentinel
        assert diagnosis == default_value
        assert identifier._report_type_tree == expected_tree
    # case: sentinel at main diagnosis level -> sentinel returned as diagnosis, tree unchanged
    for sentinel in sentinels:
        spy_llm.answers["modalitaet"] = "modality1"
        spy_llm.answers["untersuchungstyp"] = "exam_type1"
        spy_llm.answers["hauptdiagnose"] = sentinel
        modality, exam_type, diagnosis = identifier.identify_report_type("")
        assert modality == "modality1"
        assert exam_type == "exam_type1"
        assert diagnosis == sentinel
        assert identifier._report_type_tree == expected_tree


# tests for identify_report_type_multi_choice
# -  expected content prepared for requests to inference server
def test_identify_report_type_multi_choice_prompt_content():
    """
    Tests that the prompts sent to the LLM contain both the report text and the response template
    for each pipeline used by identify_report_type_multi_choice (modality and exam_type, multi_choice).
    """
    example_report = "CT des Abdomens: Leber unauffällig."
    pipeline_types = ["modality", "exam_type, multi_choice"]
    mock_llm = MockLLM({})
    mock_llm.answers = {
            "modalitaet": "modality1",
            "untersuchungstyp": ["exam_type1"]
        }
    identifier = ReportTypeIdentifier(
        "", "tests/data/initial_examples_dummy.yaml",
        categorize_novelties=False, _llm=mock_llm
    )
    # check report text appears in prompt at each classification level
    mock_llm.template_to_context.update({
        identifier._response_templates[pt].__name__: example_report
        for pt in pipeline_types
    })
    identifier.identify_report_type_multi_choice(example_report)
    # check response template appears in prompt at each classification level
    mock_llm.template_to_context.update({
        identifier._response_templates[pt].__name__: identifier._template_read_versions[pt]
        for pt in pipeline_types
    })
    identifier.identify_report_type_multi_choice(example_report)


# - content of responses received from inference server handled appropriately
def test_identify_report_type_multi_choice_existing_categories():
    """
    Tests that when the LLM responds with existing categories for modality and exam type,
    identify_report_type_multi_choice returns the LLM's selections and leaves _report_type_tree
    unchanged. Only applicable with categorize_novelties=False, as True raises ValueError.
    Covers single and multiple known exam types in the response.
    """
    initial_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    spy_llm = StubLLM()
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=False, _llm=spy_llm)
    identifier._report_type_tree = deepcopy(initial_tree)
    # case: single existing exam type
    spy_llm.answers["modalitaet"] = "modality1"
    spy_llm.answers["untersuchungstyp"] = ["exam_type1"]
    modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == ["exam_type1"]
    assert identifier._report_type_tree == initial_tree
    # case: multiple existing exam types
    spy_llm.answers["untersuchungstyp"] = ["exam_type1", "exam_type2"]
    modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == ["exam_type1", "exam_type2"]
    assert identifier._report_type_tree == initial_tree


def test_identify_report_type_multi_choice_dynamic_tree_raises_exception():
    """
    Tests that identify_report_type_multi_choice raises ValueError when categorize_novelties=True,
    as adding novelties is incompatible with multi-choice exam type selection.
    """
    mock_llm = StubLLM()
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=True, _llm=mock_llm)
    with pytest.raises(ValueError):
        identifier.identify_report_type_multi_choice("")


def test_identify_report_type_multi_choice_static_tree_reject_novelties():
    """
    Tests that with categorize_novelties=False (required), unknown responses at each level
    return a list containing only the default value and leave the report type tree unchanged.
    Unlike identify_report_type, multi_choice filters out unrecognized exam types rather than
    short-circuiting; only when all returned exam types are unrecognized does the list fall back
    to [default_value].
    """
    mock_llm = StubLLM()
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=False, _llm=mock_llm)
    default_value = identifier._special_responses["default_field_value"]
    expected_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    identifier._report_type_tree = deepcopy(expected_tree)
    # case: unknown modality -> Pydantic rejects value (modality field uses Literal), _identify_category warns,
    # returns default_value; exam type not queried (unset key would raise KeyError), tree unchanged
    mock_llm.answers["modalitaet"] = "unknown_modality"
    with pytest.warns(UserWarning, match="LLM failed to respond in accordance with schema"):
        modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == default_value
    assert exam_types == [default_value]
    assert identifier._report_type_tree == expected_tree
    # case: only unknown exam types -> all filtered out, returns [default_value], tree unchanged
    # exam_type field uses plain list[str] (no ValidationError); no matching type found triggers its own warning
    mock_llm.answers["modalitaet"] = "modality1"
    mock_llm.answers["untersuchungstyp"] = ["unknown_type1", "unknown_type2"]
    with pytest.warns(UserWarning, match="LLM failed to provide a appropriate exam type selection responses"):
        modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == [default_value]
    assert identifier._report_type_tree == expected_tree
    # case: mix of known and unknown exam types -> only known ones kept, tree unchanged
    # at least one type matches so the empty-list branch is not reached; no warning
    mock_llm.answers["untersuchungstyp"] = ["exam_type1", "unknown_type"]
    modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == ["exam_type1"]
    assert identifier._report_type_tree == expected_tree
    # sanity check: all known exam types -> all returned, tree unchanged
    mock_llm.answers["untersuchungstyp"] = ["exam_type1", "exam_type2"]
    modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == ["exam_type1", "exam_type2"]
    assert identifier._report_type_tree == expected_tree


def test_identify_report_type_multi_choice_sentinel_responses():
    """
    Tests sentinel response handling in identify_report_type_multi_choice:
    - Sentinel at modality level: exam type not queried, returns (sentinel, [default_value]).
    - other at position 0 in exam type list: returns (modality, [other]).
    """
    spy_llm = StubLLM()
    identifier = ReportTypeIdentifier("", "tests/data/initial_examples_dummy.yaml",
                                      categorize_novelties=False, _llm=spy_llm)
    default_value = identifier._special_responses["default_field_value"]
    no_findings = identifier._special_responses["no_findings_response"]
    other = identifier._special_responses["other_response"]
    multiple_exams = identifier._special_responses["multiple_exams_response"]
    sentinels = [no_findings, other, multiple_exams, default_value]
    expected_tree = {
        "modality1": {"exam_type1": ["diagnosis1", "diagnosis2"], "exam_type2": ["diagnosis1"]},
        "modality2": {"exam_type1": ["diagnosis1"]}
    }
    identifier._report_type_tree = deepcopy(expected_tree)
    # case: sentinel at modality level -> exam type not queried (unset key would raise KeyError),
    # returns (sentinel, [default_value]), tree unchanged
    for sentinel in sentinels:
        spy_llm.answers["modalitaet"] = sentinel
        modality, exam_types = identifier.identify_report_type_multi_choice("")
        assert modality == sentinel
        assert exam_types == [default_value]
        assert identifier._report_type_tree == expected_tree
    # case: other at position 0 in exam type list -> returns (modality, [other]), tree unchanged
    spy_llm.answers["modalitaet"] = "modality1"
    spy_llm.answers["untersuchungstyp"] = [other]
    modality, exam_types = identifier.identify_report_type_multi_choice("")
    assert modality == "modality1"
    assert exam_types == [other]
    assert identifier._report_type_tree == expected_tree
