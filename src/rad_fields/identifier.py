from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from pydantic import create_model, ValidationError, SecretStr
import yaml
import re
from typing import Literal, Any  # required for eval call to dynamically create response template for modality classification
from json import JSONDecodeError
from warnings import warn
from copy import copy

from .setup import PROMPT_TEMPLATES, SPECIAL_RESPONSES


class ReportTypeIdentifier:
    def __init__(self, model_name: str, initialization_file_path: str | None = None, raw_report_type_tree: dict[str, Any] | None = None,
                 categorize_novelties: bool = True, api: Literal["ollama", "openai"] = "ollama", base_url: str | None = None,
                 api_key: SecretStr | None = None, reasoning_effort: str = "low", _llm: object = None) -> None:
        """
        :param model_name: name of LLM to use
        :param initialization_file_path: path to YAML file with all possible examination modalities (note: the set of modalities is NOT adapted by the LLM)
        & examples for possible exam types & main diagnoses.
        Expected format:
        ---
        1st modality:
            1st exam for modality:
                - 1st main diagnosis for 1st modality and 1st exam type
                ...
            2nd exam type for modality:
                ...
        2nd modality:
            ...
        ...
        :param raw_report_type_tree: nested dictionary detailing modalities, modalities & (initial) exam types or modalities,
        (initial) exam types and (initial) main diagnoses
        :param categorize_novelties: whether LLM should invent new types to label report texts,
        which don't fit in any of the categories detailed in the initialization file
        :param api: API to use to communicate with inference server
        :param base_url: base url of OpenAI-compatible server
        :param api_key: api key for OpenAI-compatible server
        :param reasoning_effort: reasoning effort parameter for requests send to OpenAI-compatible server
        """
        # raise exceptions if missing user arguments
        if api == "openai":
            if base_url is None or api_key is None:
                raise ValueError("Must provide \"base_url\" and \"api_key\" arguments if OpenAI-API is used.")
        if initialization_file_path is None:
            if raw_report_type_tree is None:
                raise ValueError("Must provide either (initial) report type tree or path to initialization file")
        # initialize raw report tree from provided YAML file
        else:
            raw_report_type_tree = yaml.safe_load(open(initialization_file_path))
        # initialize tree of possible report types from raw tree
        if isinstance(raw_report_type_tree, list):
            loaded_tree = {modality: {} for modality in raw_report_type_tree}
            warn("Interpreting provided initial report types as enumeration of exam modalities. "
                 "We strongly recommend to provide initial examples for exam types / main diagnoses.")
        elif isinstance(raw_report_type_tree, dict):
            loaded_tree = dict()
            for modality in raw_report_type_tree.keys():
                if isinstance(raw_report_type_tree[modality], str):
                    example_exam_type = raw_report_type_tree[modality]
                    loaded_tree[modality] = {example_exam_type: []}
                elif isinstance(raw_report_type_tree[modality], list):  # assumption: user provided multiple exam types without example subtypes
                    example_exam_types = raw_report_type_tree[modality]
                    loaded_tree[modality] = {exam_type: [] for exam_type in example_exam_types}
                elif isinstance(raw_report_type_tree[modality], dict):  # assumption: user provided multiple exam types with example subtypes
                    loaded_tree[modality] = dict()
                    subtypes_for_exam_types = raw_report_type_tree[modality]
                    for exam_type in subtypes_for_exam_types.keys():
                        if isinstance(raw_report_type_tree[modality][exam_type], str):
                            subtype = subtypes_for_exam_types[exam_type]
                            loaded_tree[modality][exam_type] = [subtype, ]
                        elif isinstance(raw_report_type_tree[modality][exam_type], list):
                            subtypes = subtypes_for_exam_types[exam_type]
                            loaded_tree[modality][exam_type] = subtypes
                        else:
                            loaded_tree[modality][exam_type] = []
                else:
                    raise TypeError("Initialization file / raw_report_type_tree does not define a set of modalities (An initial set of modalities "
                                    "is required, because the LLM may not add any new modalities).")
        else:  # raw_report_type_tree explicitly provided, but not correct type
            raise TypeError(f"Expected \"raw_report_type_tree\" to be either of type \"list\" or \"dict\", "
                            f"got \"{type(raw_report_type_tree)}\" instead.")
        self._report_type_tree = loaded_tree
        self.modalities = list(self._report_type_tree.keys())  # unchanged by LLM responses
        # track whether LLM should invent new labels for novelties
        self._categorize_novelties = categorize_novelties
        # create pipeline objects
        # create / assign LLM object
        if _llm is None:
            if api == "ollama":
                llm = ChatOllama(model=model_name)
            elif api == "openai":
                if base_url is None or api_key is None:
                    raise ValueError("Must set both \"api_key\" and \"base_url\" arguments to communicate with the OpenAI-compatible server.")
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, reasoning_effort=reasoning_effort)
            else:
                raise ValueError(f"Unsupported API \"{api}\". Supported backends are \"ollama\" and \"openai\".")
        else:  # replace interaction with inference backend for testing purposes
            llm = _llm
        # load / adapt prompt templates
        prompt_templates = copy(PROMPT_TEMPLATES["identifier"])  # only key value mapping changed
        if categorize_novelties:
            # replace standard prompt templates for exam type & main diagnosis with version,
            # that asks LLM to invent new labels for novelties
            for key in ["exam_type", "main_diagnosis"]:
                prompt_templates[key] = prompt_templates[f"{key}, categorize_novelties"]
        # create response templates
        self._special_responses = SPECIAL_RESPONSES
        default = self._special_responses["default_field_value"]
        # Using eval() to create Literal types dynamically from trusted internal values
        self._response_templates = {
            "modality": create_model("ModalitaetsAngabe", modalitaet=(eval(f"Literal{self.modalities + list(self._special_responses.values())}"),
                                                                      default)),
            "exam_type": create_model("UntersuchungstypAngabe", untersuchungstyp=(str, default)),
            "exam_type, multi_choice": create_model("UntersuchungstypAngabe", untersuchungstyp=(list[str], [default])),
            "main_diagnosis": create_model("HauptdiagnoseAngabe", hauptdiagnose=(str, default)),
            "icd_10": create_model("DiagnoseICD10Angabe", hauptdiagnose=(str, default), icd_10=(str, default))
        }
        pipeline_types = list(self._response_templates.keys())
        # need to set additional kwarg for ollama API to use guided generation for structured output instead of tool calling
        self._pipelines = {
            pipeline_type:
            # request response structured according to template & retried up to 3 times,
            # if response string can not be parsed / fails validation
                (prompt_templates[pipeline_type] | llm.with_structured_output(schema=self._response_templates[pipeline_type], method="json_schema")
                 .with_retry(retry_if_exception_type=(ValidationError, JSONDecodeError)))
            for pipeline_type in pipeline_types
        }
        self._template_read_versions = {
            pipeline_type: self._response_templates[pipeline_type]().model_dump_json() for pipeline_type in pipeline_types
        }

    def get_known_exam_types(self, modality: str) -> list[str]:
        """
        Get known exam types for a given modality.

        :param modality: The modality to query
        :return: List of known exam types for the modality
        :raises KeyError: If modality is not found
        """
        return list(self._report_type_tree[modality].keys())

    def get_known_main_diagnoses(self, modality: str, exam_type: str) -> list[str]:
        """
        Get known main diagnoses for a given modality and exam type.

        :param modality: The modality to query
        :param exam_type: The exam type to query
        :return: List of known main diagnoses
        :raises KeyError: If modality or exam_type is not found
        """
        return self._report_type_tree[modality][exam_type]

    def export_report_type_tree(self, save_file_path: str):
        """
        Export the report type tree to a YAML file.

        :param save_file_path: Path where the YAML file should be saved
        """
        yaml.safe_dump(self._report_type_tree, open(save_file_path, 'w'), allow_unicode=True)

    def _identify_category(self, report_text: str, pipeline_type: str, categories: list[str]) -> str | list[str]:
        """
        Handles structured output generation with retries.

        :param report_text: The radiology report text to analyze
        :param pipeline_type: One of "modality", "exam_type", "exam_type, multi_choice", "main_diagnosis"
        :param categories: Category options for LLM to choose from
        :return: Category or list of categories returned by LLM
        """
        # get pipeline, response template & read version of template
        pipeline = self._pipelines[pipeline_type]
        response_template = self._response_templates[pipeline_type]
        template_read_version = self._template_read_versions[pipeline_type]
        # compile required information to "fill in" prompt template
        prompt_template_args = dict(
            categories=categories,
            template_read_version=template_read_version,
            report_text=report_text,
            **self._special_responses
        )
        # send request to server & extract information from response
        try:
            response_model_instance = pipeline.invoke(prompt_template_args)  # pydantic model instance returned by LLM with structured output
            category = response_model_instance.model_dump()[list(response_template.model_fields.keys())[0]]  # response model has only a single field
        except (ValidationError, JSONDecodeError):  # no valid response
            warn("LLM failed to respond in accordance with schema after two retries.")
            category = self._special_responses["default_field_value"]
        return category

    @staticmethod
    def _match(raw_category: str, known_categories: list[str]) -> tuple[str, bool]:
        """
        Tries to match raw_category with an element of known_categories using regex search.
        In case of multiple matches, returns match with the most characters.

        :param raw_category: The raw category string to match
        :param known_categories: List of known category strings to match against
        :return: Tuple of (matching category, True) if found, or (raw_category, False) otherwise
        """
        # handle empty string category (matches any string)
        if "" in known_categories:
            if raw_category == "":
                return raw_category, True
            known_categories = [cat for cat in known_categories if cat != ""]
        # try match with other categories using regex
        matching_category = None
        for category in known_categories:
            # convert category to regex by escaping all special chars
            if re.search(re.escape(category), raw_category):
                if matching_category is None or len(matching_category) < len(category):
                    matching_category = category
        if matching_category is None:
            return raw_category, False
        else:
            return matching_category, True

    def identify_report_type(self, report_text: str, extract_main_diagnosis: bool = True) -> tuple[str, str, str] | tuple[str, str]:
        """
        Identifies the type of report_text (exam modality, exam type & main diagnosis) and updates known exam types
        and main diagnoses if a new type was identified.

        :param report_text: The radiology report text to identify
        :param extract_main_diagnosis: Whether to extract the main diagnosis from the report text (default: True)
        :return: Tuple of (exam modality, exam type, main diagnosis) if extract_main_diagnosis is True,
                 otherwise (exam modality, exam type). If the LLM returns a special response (e.g. no findings,
                 multiple exams, unrecognised category) for modality or exam type, the remaining fields are set
                 to the default field value (empty string), the classification / generation process is terminated prematurely.
        """
        # classify according to modality
        modality_response = self._identify_category(report_text, "modality", self.modalities)
        if modality_response in self._report_type_tree.keys():
            known_exam_types = list(self._report_type_tree[modality_response].keys())
        else:  # LLM did not assign a modality -> special response
            if extract_main_diagnosis:
                return modality_response, self._special_responses["default_field_value"], self._special_responses["default_field_value"]
            else:
                return modality_response, self._special_responses["default_field_value"]
        # identify exam type
        exam_type_response = self._identify_category(report_text, "exam_type", known_exam_types)
        # classify exam type if possible
        exam_type_response, matches_type = self._match(exam_type_response, known_exam_types)
        if not matches_type:
            # try to find match with special response
            exam_type_response, matches_special = self._match(exam_type_response, list(self._special_responses.values()))
            if matches_special:
                if extract_main_diagnosis:
                    return modality_response, exam_type_response, self._special_responses["default_field_value"]
                else:
                    return modality_response, exam_type_response
            else:
                if self._categorize_novelties:
                    # assumption: new exam type suggested -> add new exam type
                    self._report_type_tree[modality_response][exam_type_response] = []
                else:
                    warn("LLM failed to provide an appropriate exam type selection response within the allowed number of tries.")
                    if extract_main_diagnosis:
                        return modality_response, self._special_responses["default_field_value"], self._special_responses["default_field_value"]
                    else:
                        return modality_response, self._special_responses["default_field_value"]
        # proceed to extract main diagnosis if requested
        if not extract_main_diagnosis:
            return modality_response, exam_type_response
        else:
            # identify main diagnosis
            known_diagnoses = self._report_type_tree[modality_response][exam_type_response]
            main_diagnosis_response = self._identify_category(report_text, "main_diagnosis", known_diagnoses)
            # classify diagnosis or add new diagnosis to list
            main_diagnosis_response, matches_type = self._match(main_diagnosis_response, known_diagnoses)
            if not matches_type:
                # try to find match with special response
                main_diagnosis_response, matches_special = self._match(main_diagnosis_response, list(self._special_responses.values()))
                if matches_special:
                    return modality_response, exam_type_response, main_diagnosis_response
                else:
                    if self._categorize_novelties:
                        # assumption: new main diagnosis suggested -> add new main diagnosis
                        self._report_type_tree[modality_response][exam_type_response].append(main_diagnosis_response)
                    else:
                        warn("LLM failed to provide an appropriate main diagnosis selection response within the allowed number of tries.")
                        return modality_response, exam_type_response, self._special_responses["default_field_value"]
            return modality_response, exam_type_response, main_diagnosis_response

    def identify_report_type_multi_choice(self, report_text: str) -> tuple[str, list[str]]:
        """
        Identifies the type of report_text (exam modality & possible exam types).

        :param report_text: The radiology report text to identify
        :return: Tuple of (exam modality, list of exam types). If the LLM returns a special response for modality,
                 the exam types list contains only the default field value (empty string). If no exam type response
                 matches a known type, the first response is checked against special responses; if it matches, it is
                 returned as a single-element list, otherwise the list contains only the default field value.
        """
        if self._categorize_novelties:
            raise ValueError("Adding new categories for novelties is a feature for a first database screening run, it is not supported in "
                             "combination with multiple choice selection for the examination type.")
        # classify according to modality
        modality_response = self._identify_category(report_text, "modality", self.modalities)
        if modality_response in self._report_type_tree.keys():
            known_exam_types = list(self._report_type_tree[modality_response].keys())
        else:  # LLM did not assign a modality -> special response
            return modality_response, [self._special_responses["default_field_value"]]
        # identify exam type
        exam_type_responses = self._identify_category(report_text, "exam_type, multi_choice", known_exam_types)
        # classify exam types if possible
        exam_types = []
        for exam_type_response in exam_type_responses:
            exam_type, matches_type = self._match(exam_type_response, known_exam_types)
            if matches_type:
                exam_types.append(exam_type)
        if not exam_types:
            # try to match first response with special response
            try:
                exam_type_response, matches_special = self._match(exam_type_responses[0], self._special_responses.values())
            except IndexError:  # LLM responded with empty list
                matches_special = False
            if matches_special:
                exam_types = [exam_type_response]
            else:
                warn("LLM failed to provide a appropriate exam type selection responses within the allowed number of tries.")
                exam_types = [self._special_responses["default_field_value"]]
        return modality_response, exam_types

    def infer_icd_code(self, report_text: str, return_diagnosis: bool = False) -> str | tuple[str, str]:
        """
        Identifies the main diagnosis of report_text and classifies it according to the ICD-10 guidelines.

        :param report_text: The radiology report text to analyze
        :param return_diagnosis: Whether to return the free-text diagnosis provided by the LLM (default: False)
        :return: 3-character ICD code or empty string if no valid code could be extracted.
                 If return_diagnosis is True, returns tuple of (ICD code, free-text diagnosis)
        """
        # get pipeline, response template & read version of template
        category_type = "icd_10"
        pipeline = self._pipelines[category_type]
        template_read_version = self._template_read_versions[category_type]
        prompt_template_args = {
            "template_read_version": template_read_version,
            "report_text": report_text
        }
        diagnosis_and_code = pipeline.invoke(prompt_template_args).model_dump()
        try:
            icd_code = re.search(r'[A-Z]\d{2}', diagnosis_and_code["icd_10"]).group()
        except AttributeError:
            icd_code = ""
        if not return_diagnosis:
            return icd_code
        else:
            return icd_code, diagnosis_and_code["hauptdiagnose"]
