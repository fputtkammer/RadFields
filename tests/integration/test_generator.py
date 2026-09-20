import pandas as pd
import json

from src.rad_fields.generator import StructuredReportGenerator

def test_select_template():
    """
    Tests StructuredReportGenerator.select_template(): verifies that returned modality, exam type,
    and template name values have correct types and are consistent with the template index.
    """
    # check correct output types for example report
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "llama3.1")
    # load data
    fictitious_reports_df = pd.read_csv("tests/data/befunde.csv")
    fictitious_reports = fictitious_reports_df["Befundtext"]
    # single choice selection for exam type
    for example_report in fictitious_reports:
        results = generator.select_templates(example_report)
        assert isinstance(results["modality"], str)
        assert isinstance(results["exam_types"], list)
        assert len(results["exam_types"]) == 1
        selected_exam_type = results["exam_types"][0]
        assert isinstance(selected_exam_type, str)
        assert isinstance(results["template_names"], list)
        for template_name in results["template_names"]:
            assert isinstance(template_name, str)
        # check modality, exam type & template name consistent with template index
        if results["modality"] not in generator._special_responses.values():
            assert results["modality"] in generator._template_tree.keys()
            if selected_exam_type not in generator._special_responses.values():
                assert selected_exam_type in generator._template_tree[results["modality"]].keys()
                if results["template_names"][0] not in generator._special_responses.values():
                    for template_name in results["template_names"]:
                        assert template_name in generator._template_tree[results["modality"]][selected_exam_type]


def test_select_templates_for_multiple_exams_detailed():
    # check correct output types for example report
    # load data
    fictitious_reports_df = pd.read_csv("tests/data/befunde.csv")
    # concatenated reports -> expectation multiple exam types / templates returned
    generator = StructuredReportGenerator("tests/data/templates/template_index_parent_child.yaml", "llama3.1")
    ct_reports = []
    for exam_type in ["CT Abdomen", "CT Schädel"]:
        ct_reports.append(f"{exam_type} from [Date]: \n {fictitious_reports_df[fictitious_reports_df["Untersuchungen"] == exam_type]
        ["Befundtext"].item()}")
    concatenated_report = "\n\n".join(ct_reports)
    results = generator.select_templates_for_multiple_exams_detailed(concatenated_report)
    assert results["modality"] in generator._template_tree.keys()
    assert len(results["exam_types"]) != 1
    for exam_type in results["exam_types"]:
        assert exam_type in generator._template_tree[results["modality"]].keys()
    assert len(results["template_names"]) != 1
    template_names_for_exam_types = [template_name for exam_type in results["exam_types"]
                                     for template_name in generator._template_tree[results["modality"]][exam_type]]
    for template_name in results["template_names"]:
        assert template_name in template_names_for_exam_types


def test_structure_report():
    """
    Tests StructuredReportGenerator.structure_report(): verifies that the output is a valid JSON
    string conforming to the selected template's Pydantic model, that the template reader view and
    report text appear in the LLM prompt, and that appropriate ValueError exceptions are raised for
    sentinel responses, unknown template names, and template entries missing model metadata.
    """
    # check correct output types for example report
    generator = StructuredReportGenerator("tests/data/templates/template_index_simple.yaml", "llama3.1")
    reports_df = pd.read_csv("tests/data/befunde.csv")
    example_us_report = reports_df[reports_df["Untersuchungen"] == "Sonographie Abdomen"].squeeze()["Befundtext"]
    template_name = "AbdomenTemplate"
    structured_report_text = generator.structure_report(example_us_report, [template_name])
    assert isinstance(structured_report_text, str)
    # check valid JSON string
    structured_report = json.loads(structured_report_text)
    # validate structured report
    selected_template = generator._template_data[template_name]["model"]
    selected_template(**structured_report)


def test_structure_report_sequentially():
    """
    Tests StructuredReportGenerator.structure_report_sequentially(): verifies that structuring a
    multi-exam report against multiple templates returns a list of valid JSON strings, each
    conforming to the corresponding template's Pydantic model.
    """
    # check correct output types for example report
    generator = StructuredReportGenerator("tests/data/templates/template_index_parent_child.yaml", "llama3.1")
    reports_df = pd.read_csv("tests/data/befunde.csv")
    # join two reports
    example_ct_reports = [reports_df[reports_df["Untersuchungen"] == exam].squeeze()["Befundtext"] for exam in ["CT Abdomen", "CT Schädel"]]
    example_ct_multi_exam_report = "\n".join(example_ct_reports)
    template_names = ['CtAbdomenUndBeckenStandardTemplate', 'CtKopfMitOderOhneKontrastmittelTemplate']
    structured_report_strings = generator.structure_report_sequentially(example_ct_multi_exam_report, template_names)
    for i_report, structured_report_str in enumerate(structured_report_strings):
        assert isinstance(structured_report_str, str)
        # check valid JSON string
        structured_report = json.loads(structured_report_str)
        # validate structured report
        selected_template = generator._template_data[template_names[i_report]]["model"]
        selected_template(**structured_report)