import yaml
import os
from pydantic import BaseModel
import pandas as pd
import json
import pytest
from langchain_core.messages import AIMessage

from src.rad_fields.generator import StructuredReportGenerator
from src.rad_fields.identifier import ReportTypeIdentifier
from src.rad_fields.utils import to_template_name, non_composite_to_string, instructions_to_pydantic_model
from tests.unit.doubles import MockLLM, StubLLM, SequentialMockLLM

# tests for __init__
def test_init_basic_template_index():
    """
    Tests StructuredReportGenerator initialization: verifies that the type tree is extracted from
    the template index and passed to the identifier, that all template names in the tree have
    associated data entries with correct types, that every template file in the directory is
    registered, and that combined exam types are added correctly with the right template counts.
    """
    # check type tree extracted correctly & passed to _identifier attribute
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    template_index = yaml.safe_load(open(template_index_path))
    expected_type_tree = {modality: {template_index_for_exam_type["Type"]: [] for template_index_for_exam_type in template_index[modality]}
                          for modality in template_index.keys()}
    assert generator._identifier._report_type_tree == expected_type_tree
    # check all template names listed in template tree have associated data entry
    template_names = [template_name for modality in generator._template_tree.keys()
                      for exam_type in generator._template_tree[modality].keys()
                      for template_name in generator._template_tree[modality][exam_type]]
    assert template_names == list(generator._template_data.keys())
    # check all entries in template data have correct types
    for template_name in template_names:
        assert isinstance(generator._template_data[template_name]["explanation"], str)
        assert issubclass(generator._template_data[template_name]["model"], BaseModel)
        assert isinstance(generator._template_data[template_name]["reader_view"], str)
    # check that every template file in templates directory has an associated entry in template tree
    template_dir = "tests/data/templates/ultrasound"
    n_template_files = len([os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(template_dir)
                            for filename in filenames if filename.endswith(".yaml")])
    assert n_template_files == len(template_names)


def test_init_combined_exams():
    """
    Tests that StructuredReportGenerator correctly loads combined exam type definitions:
    combined exam types are added to the identifier's report type tree on top of the base exam
    types, and the template pool for each combined exam type is the deduplicated union of its
    constituent base exam types' templates.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    template_index = yaml.safe_load(open(template_index_path))
    expected_type_tree = {modality: {template_index_for_exam_type["Type"]: [] for template_index_for_exam_type in template_index[modality]}
                          for modality in template_index.keys()}
    combined_exams_path = "tests/data/templates/combined_exam_definitions.yaml"
    modality = "Ultraschall"  # simplified test template index considers only one modality
    generator = StructuredReportGenerator(template_index_path, "", combined_exams_path=combined_exams_path)
    # check combined exam types are added to template tree
    combined_exam_defs = yaml.safe_load(open(combined_exams_path))
    n_combined_exam_types = len(combined_exam_defs[modality])
    n_expected_exam_types = n_combined_exam_types + len(expected_type_tree[modality].keys())
    assert n_expected_exam_types == len(generator._identifier._report_type_tree[modality].keys())
    # check number of templates per combined exam type is correct
    for exam_type_def in combined_exam_defs[modality]:
        combined_exam_name = exam_type_def["Type"]
        expected_base_exams = exam_type_def["Definition"]["combines"]
        n_expected_templates = len(set([template_name for base_exam in expected_base_exams
                                        for template_name in generator._template_tree[modality][base_exam]]))
        assert n_expected_templates == len(generator._template_tree[modality][combined_exam_name])


def test_init_parent_child_logic():
    """
    tests whether parent / child template entries in template index are handled correctly when initializing StructuredReportGenerator
    :return:
    """
    # check type tree extracted correctly & passed to _identifier attribute
    template_index_path = "tests/data/templates/template_index_parent_child.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    template_index = yaml.safe_load(open(template_index_path))
    expected_type_tree = {modality: {template_index_for_exam_type["Type"]: [] for template_index_for_exam_type in template_index[modality]}
                          for modality in template_index.keys()}
    assert generator._identifier._report_type_tree == expected_type_tree
    # check all template names listed in template tree have associated data entry
    template_names = [template_name for modality in generator._template_tree.keys()
                      for exam_type in generator._template_tree[modality].keys()
                      for template_name in generator._template_tree[modality][exam_type]]
    assert set(template_names) == set(generator._template_data.keys())
    # check all entries in template data have correct types
    for template_name in template_names:
        assert isinstance(generator._template_data[template_name]["explanation"], str)
        assert issubclass(generator._template_data[template_name]["model"], BaseModel)
        assert isinstance(generator._template_data[template_name]["reader_view"], str)
    # check that every template file in templates directory has at least one associated entry in template tree
    template_files_dir = "tests/data/templates/ct"
    n_template_files = len(["" for dirpath, dirnames, filenames in os.walk(template_files_dir)
                            for filename in filenames if filename.endswith(".yaml")])
    assert n_template_files <= len(set(template_names))
    # check instructions merged correctly
    template_dir = "tests/data/templates"
    for modality in template_index.keys():
        for template_index_for_exam_type in template_index[modality]:
            template_entries = template_index_for_exam_type["Templates"]
            for template_entry in template_entries:
                template_name = to_template_name(template_entry["name"])
                parent_entry = template_entry["parent"]
                template_filepath = os.path.join(template_dir, template_entry["filepath"])
                template_instructions = yaml.safe_load(open(template_filepath))
                reader_view = generator._template_data[template_name]["reader_view"]
                pydantic_model_from_db = generator._template_data[template_name]["model"]
                model_title = pydantic_model_from_db.__name__
                model_schema_from_instructions = instructions_to_pydantic_model(non_composite_to_string(template_instructions),
                                                                                model_title)[0].model_json_schema()
                if not parent_entry:  # parent or independent template
                    # check reader-view same as instructions in file
                    assert reader_view == json.dumps(template_instructions)
                    # check model as implied by instructions in file
                    assert pydantic_model_from_db.model_json_schema() == model_schema_from_instructions
                else:  # child template
                    # check reader-view different from instructions in file
                    assert reader_view != json.dumps(template_instructions)
                    # check model not as implied by instructions in file (result of merged instructions for parent & child template)
                    assert pydantic_model_from_db.model_json_schema() != model_schema_from_instructions


def test_init_erroneous_template_index(tmp_path):
    """
    Tests that StructuredReportGenerator.__init__() raises ValueError with appropriate messages
    for erroneous template index files:
    - a template is assigned both parent and children roles
    - a child is claimed by two different parent templates
    - a child's declared parent conflicts with the parent claiming it
    - a child template is never claimed by any parent
    - a template file contains invalid YAML
    - a template file referenced in the index does not exist
    - a template index entry is missing a required key
    - a child template has an incompatible type definition relative to its parent
    - a template name appears more than once in the index
    """
    index_file = tmp_path / "index.yaml"
    valid_template = tmp_path / "valid_template.yaml"
    valid_template.write_text("befund: ''\n")

    # Case 1: template defined as both parent and child
    # ValueError raised before file is read, so filepath need not exist
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: CT Anderes Template\n"
        "        children:\n"
        "          - CT Kopf Kind\n"
        "        filepath: irrelevant.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="multiple roles \\(parent and child\\)"):
        StructuredReportGenerator(str(index_file), "")

    # Case 2: child claimed by two different parents
    # The first parent's file is read before the second parent's conflict is detected
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children:\n"
        "          - CT Kopf Kind\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Test\n"
        "      - name: CT Kopf Advanced\n"
        "        parent: false\n"
        "        children:\n"
        "          - CT Kopf Kind\n"
        "        filepath: irrelevant.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="two parents"):
        StructuredReportGenerator(str(index_file), "")

    # Case 3: child declares a different parent than the one claiming it
    # The first parent's file is read before the child's conflict is detected
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children:\n"
        "          - CT Kopf Kind\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Test\n"
        "      - name: CT Kopf Kind\n"
        "        parent: CT Anderes Template\n"
        "        children: false\n"
        "        filepath: irrelevant.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="conflicting parents"):
        StructuredReportGenerator(str(index_file), "")

    # Case 4: child references a parent that never claims it
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Kind\n"
        "        parent: CT Kopf Standard\n"
        "        children: false\n"
        "        filepath: irrelevant.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="not explicitly referenced by any parent"):
        StructuredReportGenerator(str(index_file), "")

    # Case 5: template file exists but contains invalid YAML
    bad_template = tmp_path / "bad_template.yaml"
    bad_template.write_text(":invalid: yaml: [unclosed")
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: bad_template.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="Error for template at"):
        StructuredReportGenerator(str(index_file), "")

    # Case 6: referenced template file does not exist
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: nonexistent_template.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(FileNotFoundError, match="Could not find template file"):
        StructuredReportGenerator(str(index_file), "")

    # Case 7: template index entry is missing a required key
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        # "explanation" key is intentionally omitted
    )
    with pytest.raises(ValueError, match="Missing"):
        StructuredReportGenerator(str(index_file), "")

    # Case 8: child template has incompatible type definition relative to parent
    # Parent defines "befund" as a nested dict; child redefines it as a list — triggers TypeError in extend_instructions
    parent_template = tmp_path / "parent_template.yaml"
    parent_template.write_text("befund:\n  leber: ''\n")
    child_template = tmp_path / "child_template.yaml"
    child_template.write_text("befund:\n  - leber: ''\n")
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children:\n"
        "          - CT Kopf Kind\n"
        "        filepath: parent_template.yaml\n"
        "        explanation: Test\n"
        "      - name: CT Kopf Kind\n"
        "        parent: CT Kopf Standard\n"
        "        children: false\n"
        "        filepath: child_template.yaml\n"
        "        explanation: Test\n"
    )
    with pytest.raises(ValueError, match="Could not update parent template definition for child template"):
        StructuredReportGenerator(str(index_file), "")

    # Case 9: duplicate template name in the index
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Test\n"
        "      - name: CT Kopf Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Duplicate\n"
    )
    with pytest.raises(ValueError, match="multiple template index entries"):
        StructuredReportGenerator(str(index_file), "")

    # Case 10: same template name appearing in two different exam types
    # Redefinition with different data (different explanation) raises ValueError.
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Erste Definition\n"
        "  - Type: CT Abdomen\n"
        "    Templates:\n"
        "      - name: CT Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Zweite Definition\n"
    )
    with pytest.raises(ValueError, match="redefines an existing template with different data"):
        StructuredReportGenerator(str(index_file), "")

    # Redefinition with identical data (same explanation and same template file) is allowed:
    # the template is registered under both exam types without error.
    index_file.write_text(
        "Computer Tomographie:\n"
        "  - Type: CT Kopf\n"
        "    Templates:\n"
        "      - name: CT Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Gleiche Definition\n"
        "  - Type: CT Abdomen\n"
        "    Templates:\n"
        "      - name: CT Standard\n"
        "        parent: false\n"
        "        children: false\n"
        "        filepath: valid_template.yaml\n"
        "        explanation: Gleiche Definition\n"
    )
    generator = StructuredReportGenerator(str(index_file), "")
    duplicate_template_name = to_template_name("CT Standard")
    assert duplicate_template_name in generator._template_tree["Computer Tomographie"]["CT Kopf"]
    assert duplicate_template_name in generator._template_tree["Computer Tomographie"]["CT Abdomen"]
    assert generator._template_data[duplicate_template_name]["explanation"] == "Gleiche Definition"

# tests for select_template method
# - identification delegated to _identifier
def test_select_template_delegates_to_identifier():
    """
    Tests that select_template delegates modality and exam type identification to
    _identifier.identify_report_type (not identify_report_type_multi_choice):
    - identify_report_type is called exactly once with the provided report text and extract_main_diagnosis=False
    - the modality and exam type it returns are reflected in the result
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    example_report = "Sonographie des Abdomens: Leber unauffällig."
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_type = list(generator._template_tree[modality].keys())[0]

    calls = []
    def spy_identify_report_type(report_text, extract_main_diagnosis=True):
        calls.append({"report_text": report_text, "extract_main_diagnosis": extract_main_diagnosis})
        return modality, exam_type

    generator._identifier.identify_report_type = spy_identify_report_type
    generator._llm = MockLLM({})

    result = generator.select_templates(example_report)

    assert len(calls) == 1
    assert calls[0]["report_text"] == example_report
    assert calls[0]["extract_main_diagnosis"] == False
    assert result["modality"] == modality
    assert result["exam_types"] == [exam_type]

# -  expected content prepared for requests to inference server
def test_select_template_prompt_content():
    """
    Tests that the prompt sent to the LLM for template selection in select_template contains both
    the report text and the template names/explanations (options_reader_view), for a known
    modality and exam type returned by the identifier.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    example_report = "Sonographie des Abdomens: Leber unauffällig."
    modality = "Ultraschall"
    exam_type = "US_Abdomen"
    generator = StructuredReportGenerator(template_index_path, "")
    # replace identifier with one backed by a SpyLLM returning known modality/exam_type
    spy_llm = StubLLM()
    spy_llm.answers["modalitaet"] = modality
    spy_llm.answers["untersuchungstyp"] = exam_type
    exam_types_for_modality = {m: list(generator._template_tree[m].keys()) for m in generator._template_tree.keys()}
    generator._identifier = ReportTypeIdentifier("", raw_report_type_tree=exam_types_for_modality,
                                                 categorize_novelties=False, _llm=spy_llm)
    # build expected options_reader_view as constructed by select_template
    template_names = generator._template_tree[modality][exam_type]
    options_reader_view = "\n".join([f"\"{name}\": \"{generator._template_data[name]['explanation']}\""
                                     for name in template_names])
    # response template name for single-choice template selection
    response_template_name = "AusgewaehltesTemplateAngabe"
    mock_llm = MockLLM({})
    generator._llm = mock_llm
    # check report text appears in prompt
    mock_llm.template_to_context = {response_template_name: example_report}
    generator.select_templates(example_report)
    # check options_reader_view appears in prompt
    mock_llm.template_to_context = {response_template_name: options_reader_view}
    generator.select_templates(example_report)

# - content of responses received from inference server handled appropriately
def test_select_template_classification_response():
    """
    Tests that when identify_report_type returns an existing modality and exam type, and the LLM
    selects a valid template name, select_template returns all three values unchanged.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_type = list(generator._template_tree[modality].keys())[0]
    template_name = generator._template_tree[modality][exam_type][0]

    generator._identifier.identify_report_type = lambda report_text, extract_main_diagnosis=True: (modality, exam_type)

    spy_llm = StubLLM()
    spy_llm.answers["template_name"] = template_name
    generator._llm = spy_llm

    result = generator.select_templates("")

    assert result["modality"] == modality
    assert result["exam_types"] == [exam_type]
    assert result["template_names"] == [template_name]

def test_select_template_invalid_response():
    """
    Tests that when the LLM returns a template name not in the selection options,
    select_template returns the default field value in template_names and issues a UserWarning.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_type = list(generator._template_tree[modality].keys())[0]
    default_value = generator._special_responses["default_field_value"]

    generator._identifier.identify_report_type = lambda report_text, extract_main_diagnosis=True: (modality, exam_type)

    spy_llm = StubLLM()
    spy_llm.answers["template_name"] = "invalid_template_name"
    generator._llm = spy_llm

    with pytest.warns(UserWarning, match="LLM failed to provide an appropriate template selection response"):
        result = generator.select_templates("")

    assert result["modality"] == modality
    assert result["exam_types"] == [exam_type]
    assert result["template_names"] == [default_value]

def test_select_template_sentinel_responses():
    """
    Tests that when the identifier returns a sentinel value at the modality or exam type level,
    select_template handles them appropriately:
    - no findings / other / default value at modality / exam type level: returns sentinel as modality / exam type, default for remaining return values.
    - multiple_exams_response at exam type level: delegates to select_templates_for_multiple_exams_detailed,
      which performs multi-choice exam type and template selection; its result is passed through unchanged.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    # pick a valid modality and exam type from the generator's template tree
    modality = list(generator._template_tree.keys())[0]
    exam_types = list(generator._template_tree[modality].keys())
    # replace _identifier with a new identifier backed by SpyLLM
    spy_llm = StubLLM()
    exam_types_for_modality = {m: list(generator._template_tree[m].keys()) for m in generator._template_tree.keys()}
    generator._identifier = ReportTypeIdentifier("", raw_report_type_tree=exam_types_for_modality,
                                                  categorize_novelties=False, _llm=spy_llm)
    default_value = generator._special_responses["default_field_value"]
    no_findings = generator._special_responses["no_findings_response"]
    other = generator._special_responses["other_response"]
    multiple_exams = generator._special_responses["multiple_exams_response"]
    for sentinel in [no_findings, other, default_value]:
        # case: no_findings or other at modality level -> exam type not queried (unset key would raise KeyError),
        # modality returns sentinel, exam_types and template_names fall back to default
        spy_llm.answers["modalitaet"] = sentinel
        result = generator.select_templates("")
        assert result["modality"] == sentinel
        assert result["exam_types"] == [default_value]
        assert result["template_names"] == [default_value]
        # case: no_findings or other at exam type level -> template name not queried (unset key would raise KeyError),
        # sentinel returned as exam type, template_names default
        spy_llm.answers["modalitaet"] = modality
        spy_llm.answers["untersuchungstyp"] = sentinel
        result = generator.select_templates("")
        assert result["modality"] == modality
        assert result["exam_types"] == [sentinel]
        assert result["template_names"] == [default_value]
    # case: multiple_exams_response at exam type level -> select_templates_for_multiple_exams_detailed
    # is called to perform multi-choice exam type and template selection. Provide valid multi-choice
    # answers (two exam types, one template per exam type) as example answers for the SpyLLM
    # and verify they are returned by select_template.
    spy_llm.answers["untersuchungstyp"] = multiple_exams
    valid_exam_types = exam_types[:2]
    valid_template_names = [generator._template_tree[modality][et][0] for et in valid_exam_types]
    expected_multi_exam_result = {
        "modality": modality,
        "exam_types": valid_exam_types,
        "template_names": valid_template_names
    }
    # mock the delegated method to return the valid multi-choice results
    generator.select_templates_for_multiple_exams_detailed = lambda report_text: expected_multi_exam_result
    result = generator.select_templates("")
    assert result == expected_multi_exam_result

# tests for select_templates_for_multiple_exams_detailed method
# - identification delegated to _identifier
def test_select_templates_for_multiple_exams_detailed_delegates_to_identifier():
    """
    Tests that select_templates_for_multiple_exams_detailed delegates modality and exam type
    identification to _identifier.identify_report_type_multi_choice (not identify_report_type):
    - identify_report_type_multi_choice is called exactly once with the provided report text
    - the modality and exam types it returns are reflected in the result
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    example_report = "Sonographie des Halses und der Weichteile: Schilddrüse unauffällig, Mamma unauffällig."
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_types = list(generator._template_tree[modality].keys())[:2]

    calls = []
    def spy_identify_report_type_multi_choice(report_text):
        calls.append({"report_text": report_text})
        return modality, exam_types

    generator._identifier.identify_report_type_multi_choice = spy_identify_report_type_multi_choice
    generator._llm = MockLLM({})

    result = generator.select_templates_for_multiple_exams_detailed(example_report)

    assert len(calls) == 1
    assert calls[0]["report_text"] == example_report
    assert result["modality"] == modality
    assert result["exam_types"] == exam_types

# -  expected content prepared for requests to inference server
def test_select_templates_for_multiple_exams_detailed_prompt_content():
    """
    Tests that the prompt sent to the LLM for template selection in
    select_templates_for_multiple_exams_detailed contains both the report text and the template
    names/explanations (options_reader_view), for known modality and exam types returned by the
    identifier.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    example_report = "Sonographie des Halses und der Weichteile: Schilddrüse unauffällig, Mamma unauffällig."
    modality = "Ultraschall"
    exam_types = ["US_Hals", "US_Weichteile"]
    generator = StructuredReportGenerator(template_index_path, "")
    # replace identifier with one backed by a SpyLLM returning known modality/exam_types (list for multi-choice)
    spy_llm = StubLLM()
    spy_llm.answers["modalitaet"] = modality
    spy_llm.answers["untersuchungstyp"] = exam_types
    exam_types_for_modality = {m: list(generator._template_tree[m].keys()) for m in generator._template_tree.keys()}
    generator._identifier = ReportTypeIdentifier("", raw_report_type_tree=exam_types_for_modality,
                                                 categorize_novelties=False, _llm=spy_llm)
    # build expected options_reader_view as constructed by select_templates_for_multiple_exams_detailed
    template_names = list(dict.fromkeys(
        template_name for exam_type in exam_types for template_name in generator._template_tree[modality][exam_type]
    ))
    options_reader_view = "\n".join([f"\"{name}\": \"{generator._template_data[name]['explanation']}\""
                                     for name in template_names])
    # response template name for multi-choice template selection in this method
    response_template_name = "AusgewaehltesTemplateAngabe"
    mock_llm = MockLLM({})
    generator._llm = mock_llm
    # check report text appears in prompt
    mock_llm.template_to_context = {response_template_name: example_report}
    generator.select_templates_for_multiple_exams_detailed(example_report)
    # check options_reader_view appears in prompt
    mock_llm.template_to_context = {response_template_name: options_reader_view}
    generator.select_templates_for_multiple_exams_detailed(example_report)

# - content of responses received from inference server handled appropriately
def test_select_templates_for_multiple_exams_detailed_classification_response():
    """
    Tests that when identify_report_type_multi_choice returns existing modality and exam types, and
    the LLM selects valid template names, select_templates_for_multiple_exams_detailed returns all
    values unchanged.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_types = list(generator._template_tree[modality].keys())[:2]
    template_names = [generator._template_tree[modality][et][0] for et in exam_types]

    generator._identifier.identify_report_type_multi_choice = lambda report_text: (modality, exam_types)

    spy_llm = StubLLM()
    spy_llm.answers["template_names"] = template_names
    generator._llm = spy_llm

    result = generator.select_templates_for_multiple_exams_detailed("")

    assert result["modality"] == modality
    assert result["exam_types"] == exam_types
    assert result["template_names"] == template_names

def test_select_templates_for_multiple_exams_detailed_invalid_response():
    """
    Tests that when the LLM returns template names not in the selection options,
    select_templates_for_multiple_exams_detailed returns the default field value in template_names
    and issues a UserWarning.
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    exam_types = list(generator._template_tree[modality].keys())[:2]
    default_value = generator._special_responses["default_field_value"]

    generator._identifier.identify_report_type_multi_choice = lambda report_text: (modality, exam_types)

    spy_llm = StubLLM()
    spy_llm.answers["template_names"] = ["invalid_template_name"]
    generator._llm = spy_llm

    with pytest.warns(UserWarning, match="LLM failed to provide an appropriate template selection response"):
        result = generator.select_templates_for_multiple_exams_detailed("")

    assert result["modality"] == modality
    assert result["exam_types"] == exam_types
    assert result["template_names"] == [default_value]

def test_select_templates_for_multiple_exams_detailed_sentinel_responses():
    """
    Tests that when the identifier returns a sentinel value at the modality or exam type level,
    select_templates_for_multiple_exams_detailed handles them appropriately, mirroring
    select_templates' behavior for sentinel responses:
    - sentinel at modality level: modality returns sentinel, exam_types falls back to default,
      template_names falls back to [default_value] (early return before template selection).
    - sentinel at exam type level: multi-choice matching does not preserve unmatched responses,
      so all sentinels collapse to [default_value]; early return with template_names == [default_value].
    """
    template_index_path = "tests/data/templates/template_index_simple.yaml"
    generator = StructuredReportGenerator(template_index_path, "")
    modality = list(generator._template_tree.keys())[0]
    spy_llm = StubLLM()
    exam_types_for_modality = {m: list(generator._template_tree[m].keys()) for m in generator._template_tree.keys()}
    generator._identifier = ReportTypeIdentifier("", raw_report_type_tree=exam_types_for_modality,
                                                  categorize_novelties=False, _llm=spy_llm)
    default_value = generator._special_responses["default_field_value"]
    no_findings = generator._special_responses["no_findings_response"]
    other = generator._special_responses["other_response"]
    multiple_exams = generator._special_responses["multiple_exams_response"]
    for sentinel in [no_findings, other, default_value, multiple_exams]:
        # case: sentinel at modality level -> exam type not queried (unset key would raise KeyError),
        # modality returns sentinel, exam_types falls back to default, template_names falls back to [default_value]
        spy_llm.answers["modalitaet"] = sentinel
        result = generator.select_templates_for_multiple_exams_detailed("")
        assert result["modality"] == sentinel
        assert result["exam_types"] == [default_value]
        assert result["template_names"] == [default_value]
    # case: other at exam type level -> sentinel value preserved, template_names falls back to [default_value]
    spy_llm.answers["modalitaet"] = modality
    spy_llm.answers["untersuchungstyp"] = [other]
    result = generator.select_templates_for_multiple_exams_detailed("")
    assert result["modality"] == modality
    assert result["exam_types"] == [other]
    assert result["template_names"] == [default_value]

# tests for structure_report method
# -  user input handled appropriately
def test_structure_report_list_template_names():
    """
    Tests that structure_report handles list[str] as template_names correctly:
    - Single-element list: executes the normal single-template structuring path.
      Uses MockLLM to verify the prompt starts with the expected system message prefix
      (distinguishes structure_report from structure_report_sequentially) and that the
      return value is a valid JSON string.
    - Multi-element list: delegates to structure_report_sequentially.
      Mocks the delegated method to verify it is called with the correct arguments.
    """
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    reports_df = pd.read_csv("tests/data/befunde.csv")
    example_us_report = reports_df[reports_df["Untersuchungen"] == "Sonographie Abdomen"].squeeze()["Befundtext"]
    template_name = "AbdomenTemplate"

    # single-element list: should execute the normal single-template structuring path
    system_template_str = generator._prompt_templates["structure_report"].messages[0].prompt.template
    system_prompt_start = system_template_str.split("{")[0]
    generator._llm = MockLLM({template_name: system_prompt_start})
    result = generator.structure_report(example_us_report, [template_name])
    assert isinstance(result, str)
    json.loads(result)  # valid JSON

    # multi-element list: should delegate to structure_report_sequentially
    mock_template_names = ["mock template 1", "mock template 2"]
    expected_result = ["mock result 1", "mock result 2"]
    called_with = {}

    def mock_sequential(report_text, template_names):
        called_with["report_text"] = report_text
        called_with["template_names"] = template_names
        return expected_result

    generator.structure_report_sequentially = mock_sequential
    result = generator.structure_report(example_us_report, mock_template_names)
    assert result == expected_result
    assert called_with["report_text"] == example_us_report
    assert called_with["template_names"] == mock_template_names


def test_structure_report_invalid_template_name_raises():
    """
    Tests that structure_report raises ValueError with an appropriate message for unusable
    template names: special sentinel values (other_response, default_field_value) and an unknown
    template name not registered in _template_data.
    """
    # sample setup
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    # test appropriate exception raised for unusable template name
    with pytest.raises(ValueError, match="No appropriate template found or no template provided."):
        generator.structure_report("", [generator._special_responses["other_response"]])
    with pytest.raises(ValueError, match="No appropriate template found or no template provided."):
        generator.structure_report("", [generator._special_responses["default_field_value"]])
    unknown_template_name = "unknown template"
    with pytest.raises(ValueError, match=f"Unknown template \"{unknown_template_name}\""):
        generator.structure_report("", [unknown_template_name])
    injected_template_name_wo_model = "dummy_template"
    generator._template_data[injected_template_name_wo_model] = {
        "explanation": "",
        "entry_name": ""
    }

# -  expected content prepared for requests to inference server
def test_structure_report_prompt_content():
    """
    Tests that the prompt sent to the LLM in structure_report contains both the template reader
    view (JSON string representation of default valued template instance with comments) and the report text.
    """
    # sample setup
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    reports_df = pd.read_csv("tests/data/befunde.csv")
    example_us_report = reports_df[reports_df["Untersuchungen"] == "Sonographie Abdomen"].squeeze()["Befundtext"]
    template_name = "AbdomenTemplate"
    # check template reader view appears in prompt
    instructions = generator._template_data[template_name]["reader_view"]
    generator._llm = MockLLM({template_name: instructions})
    generator.structure_report("", [template_name])
    # check report appears in prompt
    generator._llm.template_to_context = {template_name: example_us_report}
    generator.structure_report(example_us_report, [template_name])


# - content of responses received from inference server handled appropriately
def test_structure_report_returns_json_string():
    """
    Tests that structure_report returns the LLM's structured output as a JSON string
    when the LLM returns a valid response compatible with the provided template.
    """
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    template_name = "AbdomenTemplate"
    generator._llm = MockLLM({})
    result = generator.structure_report("Befundtext", [template_name])
    assert isinstance(result, str)
    result_dict = json.loads(result)
    expected = generator._template_data[template_name]["model"]().model_dump()
    assert result_dict == expected


# test for structure_report_sequentially method
def test_structure_report_sequentially_invalid_template_names_raises():
    """
    Tests that structure_report_sequentially raises ValueError if any of the provided
    template names is a special sentinel value (indicating no appropriate template was
    found/provided for one of the detailed examinations), and also if any of the provided
    template names is unknown (not registered in _template_data).
    """
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    valid_template_names = list(generator._template_data.keys())[:2]
    for special_response in generator._special_responses.values():
        template_names = valid_template_names + [special_response]
        with pytest.raises(
            ValueError,
            match="No appropriate template found or no template provided for one of the detailed examinations."
        ):
            generator.structure_report_sequentially("", template_names)

    unknown_template_name = "unknown template"
    template_names = valid_template_names + [unknown_template_name]
    with pytest.raises(ValueError, match="Unknown selected template for one of the detailed examinations"):
        generator.structure_report_sequentially("", template_names)


# -  expected content prepared for requests to inference server
def test_structure_report_sequentially_prompt_content():
    """
    Tests that in structure_report_sequentially:
    - the system prompt contains all templates and their titles in every LLM invocation
    - user prompts reference the template to be filled-in at each step
    - conversation history (prior AI responses) is part of follow-up messages sent to the LLM
    Tested for both template_index_simple.yaml and template_index_parent_child.yaml with randomly
    sampled template names.
    """
    example_report = "Befundtext für mehrere Untersuchungen."
    for template_index_path in [
        "tests/data/templates/template_index_simple.yaml",
        "tests/data/templates/template_index_parent_child.yaml",
    ]:
        generator = StructuredReportGenerator(template_index_path, "")
        all_template_names = list(generator._template_data.keys())
        template_names = all_template_names[:3]

        recording_llm = SequentialMockLLM()
        generator._llm = recording_llm

        generator.structure_report_sequentially(example_report, template_names)

        # one invocation per template
        assert len(recording_llm.recorded_messages) == len(template_names)

        # system prompt contains all template names and reader views in every invocation
        for messages in recording_llm.recorded_messages:
            system_content = messages[0].content
            for tn in template_names:
                assert tn in system_content
                assert generator._template_data[tn]["reader_view"] in system_content

        # initial user message references the first template and contains the report text
        initial_user_content = recording_llm.recorded_messages[0][1].content
        assert template_names[0] in initial_user_content
        assert example_report in initial_user_content

        # follow-up user messages reference the correct template at each step,
        # consistent with the schema used for that invocation
        for i in range(1, len(template_names)):
            follow_up_content = recording_llm.recorded_messages[i][-1].content
            assert template_names[i] in follow_up_content
            assert recording_llm.recorded_schemas[i].__name__ == template_names[i]

        # message history is consistent: messages at step i are exactly messages at step i-1
        # plus the AI response, plus the new follow-up
        for i in range(1, len(template_names)):
            prev_contents = [m.content for m in recording_llm.recorded_messages[i - 1]]
            curr_contents = [m.content for m in recording_llm.recorded_messages[i]]
            assert curr_contents[:-2] == prev_contents
            assert curr_contents[-2] == recording_llm.recorded_responses[i - 1]

        # conversation history is part of follow-up messages:
        # at step i the message list is [sys, user_0, AI_0, user_1, ..., AI_{i-1}, user_i] → 2 + 2*i messages
        for i in range(1, len(template_names)):
            messages = recording_llm.recorded_messages[i]
            assert len(messages) == 2 + 2 * i
            # AI responses from previous turns are present at positions 2, 4, ..., 2*i
            for j in range(i):
                ai_msg = messages[2 + 2 * j]
                assert isinstance(ai_msg, AIMessage)
                assert ai_msg.content == recording_llm.recorded_responses[j]


# - content of responses received from inference server handled appropriately
def test_structure_report_sequentially_returns_json_strings():
    """
    Tests that structure_report_sequentially returns the LLM's structured outputs as a list
    of JSON strings, one per template, when the LLM returns valid responses compatible with
    the provided templates.
    """
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "")
    all_template_names = list(generator._template_data.keys())
    template_names = all_template_names[:3]
    recording_llm = SequentialMockLLM()
    generator._llm = recording_llm
    results = generator.structure_report_sequentially("Befundtext", template_names)
    assert isinstance(results, list)
    assert len(results) == len(template_names)
    for result, template_name in zip(results, template_names):
        assert isinstance(result, str)
        result_dict = json.loads(result)
        expected = generator._template_data[template_name]["model"]().model_dump()
        assert result_dict == expected
