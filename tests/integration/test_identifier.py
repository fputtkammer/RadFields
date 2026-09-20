import re
import pandas as pd
import yaml

from src.rad_fields.identifier import ReportTypeIdentifier

def test_identify_report_type():
    """
    tests type / format of return values of identify_report_type method as expected for sample setup & input
    :return:
    """
    # load data
    fictitious_reports = pd.read_csv("tests/data/befunde.csv")["Befundtext"]
    loaded_tree = yaml.safe_load(open("tests/data/initial_examples.yaml"))
    # case: LLM may add new categories for novelties
    identifier = ReportTypeIdentifier("llama3.1", "tests/data/initial_examples.yaml")
    # run inference
    for example_report_text in fictitious_reports:
        outputs = identifier.identify_report_type(example_report_text)
        assert len(outputs) == 3
        for output in outputs:
            assert isinstance(output, str)
        modality_response = outputs[0]
        assert modality_response in list(loaded_tree.keys()) + list(identifier._special_responses.values())
        outputs = identifier.identify_report_type(example_report_text, extract_main_diagnosis=False)
        assert len(outputs) == 2
        for output in outputs:
            assert isinstance(output, str)
        modality_response = outputs[0]
        assert modality_response in list(loaded_tree.keys())+list(identifier._special_responses.values())
    # case: standard open world classification
    identifier = ReportTypeIdentifier("llama3.1", "tests/data/initial_examples.yaml", categorize_novelties=False)
    # run inference
    for example_report_text in fictitious_reports:
        outputs = identifier.identify_report_type(example_report_text)
        assert len(outputs) == 3
        modality_response, exam_type_response, main_diagnosis_response = outputs
        special_responses = list(identifier._special_responses.values())
        default_value = identifier._special_responses["default_field_value"]
        if modality_response not in special_responses:
            assert modality_response in list(loaded_tree.keys())
            if exam_type_response not in special_responses:
                assert exam_type_response in list(loaded_tree[modality_response].keys())+special_responses
                assert main_diagnosis_response in list(loaded_tree[modality_response][exam_type_response])+special_responses
            else:
                assert main_diagnosis_response == default_value
        else:
            assert exam_type_response == default_value
            assert main_diagnosis_response == default_value
        outputs = identifier.identify_report_type(example_report_text, extract_main_diagnosis=False)
        assert len(outputs) == 2
        modality_response, exam_type_response = outputs
        if modality_response not in special_responses:
            assert modality_response in loaded_tree.keys()
            assert exam_type_response in list(loaded_tree[modality_response].keys())+special_responses
        else:
            assert exam_type_response == default_value

def test_infer_icd_code():
    """
    tests type / format of return values of infer_icd_code method as expected for sample setup & input
    :return:
    """
    identifier = ReportTypeIdentifier("llama3.1", "tests/data/initial_examples_dummy.yaml")
    fictitious_reports = pd.read_csv("tests/data/befunde.csv")["Befundtext"]
    for sample_report_text in fictitious_reports:
        icd_code_3char_pattern = r'^[A-Z]\d{2}$'
        icd_code = identifier.infer_icd_code(sample_report_text)
        if not icd_code == "":
            assert re.match(icd_code_3char_pattern, icd_code)
        icd_code, main_diagnosis = identifier.infer_icd_code(sample_report_text, return_diagnosis=True)
        if not icd_code == "":
            assert re.match(icd_code_3char_pattern, icd_code)
        assert isinstance(main_diagnosis, str)

def test_identify_report_type_multi_choice():
    """
    Tests type / format of return values of identify_report_type_multi_choice for sample setup & input.
    Verifies that the method returns a (modality, exam_types) pair where exam_types is a list of strings,
    each belonging to the known exam types for that modality or to the set of special responses.
    """
    # load data
    fictitious_reports = pd.read_csv("tests/data/befunde.csv")["Befundtext"]
    loaded_tree = yaml.safe_load(open("tests/data/initial_examples.yaml"))
    # case: standard open world classification
    identifier = ReportTypeIdentifier("llama3.1", "tests/data/initial_examples.yaml", categorize_novelties=False)
    # run inference
    for example_report_text in fictitious_reports:
        outputs = identifier.identify_report_type_multi_choice(example_report_text)
        assert len(outputs) == 2
        modality_response, exam_types_response = outputs
        special_responses = list(identifier._special_responses.values())
        default_value = identifier._special_responses["default_field_value"]
        if modality_response not in special_responses:
            assert modality_response in loaded_tree.keys()
            assert isinstance(exam_types_response, list)
            for exam_type_response in exam_types_response:
                assert isinstance(exam_type_response, str)
                assert exam_type_response in list(loaded_tree[modality_response].keys())+special_responses
        else:
            assert exam_types_response == [default_value]