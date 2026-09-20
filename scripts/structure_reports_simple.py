"""
Simple example for end-to-end structuring script: select a template for and structure each report in
REPORT_PATH via StructuredReportGenerator, writing results to OUTPUT_PATH.

Usage:
    python structure_reports_simple.py <model_name> <api> [base_url api_key]

    <api> is "ollama" or "openai"; base_url/api_key are required and only
    read when <api> is "openai".
"""
from sys import argv
import pandas as pd
from warnings import warn
from rad_fields.generator import StructuredReportGenerator
from tqdm import tqdm

MODEL = argv[1]  # LLM name
API = argv[2]
if API == "openai":
    BASE_URL = argv[3]
    API_KEY = argv[4]
else:
    BASE_URL = API_KEY = None
REPORT_PATH = "data/simple_reports.csv"  # path to .csv file with input reports
OUTPUT_PATH = "structured_reports.csv"  # path to output .csv file

def structure_report(report_text: str, report_id: str, generator: StructuredReportGenerator) -> dict[str: str]:
    """
    Select a template for and structure a single report, tolerating failures at either step.
    :param report_text: raw report text to structure
    :param report_id: identifier of the report, used for warning messages on failure
    :param generator: generator used to select a template for & structure the report
    :return: dict with template selection results merged with the structured report under "structured_report";
    empty dict if template selection failed, "structured_report" set to "" if structuring failed
    """
    try:
        selection_results = generator.select_templates(report_text)
    except Exception as e:
        warn(f"{e} Adding void entry for report with id \"{report_id}\".")
        return dict()
    try:
        structured_report = generator.structure_report(report_text, selection_results["template_names"])
    except Exception as e:
        warn(f"{e} Skipping structuring step for report with id \"{report_id}\".")
        structured_report = ""
    return dict(**selection_results, structured_report=structured_report)

# load required reports for inference
reports = pd.read_csv(REPORT_PATH, index_col="id")["text"]
generator = StructuredReportGenerator(template_index_path="../templates/de/template_index.yaml",
                                      model_name=MODEL,
                                      combined_exams_path="../templates/de/combined_exam_definitions.yaml",
                                      api=API, api_key=API_KEY, base_url=BASE_URL)
structuring_results = []
for report_id in tqdm(reports.index):
    results = structure_report(reports[report_id], report_id, generator)
    df_entry = dict(**results, id=report_id)
    structuring_results.append(df_entry)
pd.DataFrame(structuring_results).set_index("id").to_csv(OUTPUT_PATH)
