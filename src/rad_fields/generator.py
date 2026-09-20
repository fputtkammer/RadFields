import json
from typing import Type
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.exceptions import OutputParserException
import yaml
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from pydantic import create_model, ValidationError, BaseModel, SecretStr
from warnings import warn
import os
from json import JSONDecodeError
from typing import Literal  # used to define required type for template selection response at runtime
from collections import OrderedDict

from .identifier import ReportTypeIdentifier
from .utils import to_template_name, extend_instructions, instructions_to_pydantic_model, non_composite_to_string
from .setup import PROMPT_TEMPLATES, SPECIAL_RESPONSES


class StructuredReportGenerator:
    def __init__(self, template_index_path: str, model_name: str, combined_exams_path: str | None = None, api: Literal["ollama", "openai"] = "ollama",
                 base_url: str | None = None, api_key: SecretStr | None = None, reasoning_effort: str = "low"):
        """
        :param template_index_path: path to YAML file with template file paths & metadata for modalities & exam types.
        Expected format:
        ---
        modality 1:
          - Type: exam type 1
            Templates:
              - name: template name 1
                filepath: # relative path to template yaml file
                explanation: # explanation, used by LLM to decide on which template to use for structuring after modality & exam type selected
              - name: template name 2
                # further template entries for exam type 1
          - Type: exam type 2
            # further exam type entries for modality 1
        modality 2:
        ...
        :param model_name: name of LLM to use for report type identification & structuring (passed to inference engine)
        :param api: API to use to communicate with inference server
        :param base_url: base url of OpenAI-compatible server
        :param api_key: api key for OpenAI-compatible server
        :param reasoning_effort: reasoning effort parameter for requests send to OpenAI-compatible server
        :raises ValueError, if the template index is erroneous or a referenced template file contains an invalid template definition
        FileNotFoundError, if a template file referenced in the index can not be found
        """
        # extract index information
        template_index = yaml.safe_load(open(template_index_path))
        template_dir = os.path.join(*os.path.split(template_index_path)[:-1])
        self._template_tree = dict()  # mapping of modality & exam type to list of available templates
        self._template_data = dict()  # explanation, pydantic model, template instructions (dictionary used to create model with comments)
        # & reader-view (JSON-String representation of example filled-in template with comments) for templates
        # track parent / child structure
        child_to_parent = dict()
        unresolved_parent_template_references = dict()  # template names / index titles of templates with unresolved parent references
        for modality in template_index.keys():
            # track modality
            self._template_tree[modality] = dict()
            for template_index_for_exam_type in template_index[modality]:
                exam_type = template_index_for_exam_type["Type"]
                # track exam type
                self._template_tree[modality][exam_type] = []
                template_entries = template_index_for_exam_type["Templates"]
                for template_entry in template_entries:
                    # extract template information from index entry
                    try:
                        template_name = to_template_name(template_entry["name"])
                        template_filepath = os.path.join(template_dir, template_entry["filepath"])
                        template_explanation = template_entry["explanation"]
                        parent_entry = template_entry["parent"]
                        child_entries = template_entry["children"]
                    except KeyError as e:
                        missing_key = str(e).split("'")[1]
                        raise ValueError(
                            f"Missing \"{missing_key}\" in template entry for modality \"{modality}\" and exam type \"{exam_type}\": \"{template_entry}\"")
                    # determine parent template name (if applicable)
                    if child_entries:  # parent template
                        if parent_entry:  # parent template should have no parents
                            raise ValueError(f"Provided template index defines template \"{template_entry["name"]}\" "
                                             f"with multiple roles (parent and child).")
                        parent_template_name = None
                        for child_template_index_entry_name in child_entries:
                            child_template_name = to_template_name(child_template_index_entry_name)
                            if child_template_name in child_to_parent.keys():
                                raise ValueError(
                                    f"Provided template index defines two parents (\"{child_to_parent[child_template_name]}\" and"
                                    f" \"{template_name}\") for child template \"{child_template_name}\".")
                            child_to_parent[child_template_name] = template_name
                    elif parent_entry:  # child template
                        if child_entries:
                            raise ValueError(f"Provided template index defines template \"{template_entry["name"]}\" "
                                             f"with multiple roles (parent & child).")
                        if template_name in child_to_parent.keys():
                            parent_template_name = child_to_parent[template_name]
                        else:  # parent template not processed yet
                            # -> assumption: duplicate of entry from other exam type
                            # -> resolved earlier / later
                            if template_name not in self._template_data and template_name not in unresolved_parent_template_references.keys():
                                # to be resolved later
                                unresolved_parent_template_references[template_name] = template_filepath
                            continue
                        if parent_template_name != (mentioned_parent_template_name := to_template_name(template_entry["parent"])):
                            raise ValueError(
                                f"Provided template index defines template \"{template_entry["name"]}\" with conflicting parents:"
                                f"\"{parent_template_name}\" (claims child) and \"{mentioned_parent_template_name}\" (claimed by "
                                f"child).")
                    else:  # independent template
                        parent_template_name = None

                    # create database entry for template
                    try:
                        template_instructions = yaml.safe_load(open(template_filepath))
                        # convert all fields of non-composite type (list or pydantic model) to string
                        template_instructions = non_composite_to_string(template_instructions)
                        if parent_template_name is None:
                            full_template_instructions = template_instructions
                        else:
                            try:
                                # join instructions of parent & child templates
                                full_template_instructions = extend_instructions(json.loads(self._template_data[parent_template_name]["reader_view"]),
                                                                                 template_instructions)
                            except TypeError as e:
                                # parent template is processed beforehand -> error in child template definition
                                raise ValueError(f"Could not update parent template definition for child template \"{template_entry["name"]}\": {e}")
                        try:
                            template_model = instructions_to_pydantic_model(full_template_instructions, template_name)[0]
                        except TypeError as e:
                            raise ValueError(
                                f"Could not process template instructions for template \"{template_entry["name"]}\" at \"{template_filepath}\": {e}")
                        # add template data
                        new_reader_view = json.dumps(full_template_instructions)
                        if template_name in self._template_data and template_name not in self._template_tree[modality][exam_type]:
                            # cross-exam-type redefinition: only allowed when data is identical
                            existing = self._template_data[template_name]
                            if existing["reader_view"] != new_reader_view or existing["explanation"] != template_explanation:
                                error_message = f"Template \"{template_entry["name"]}\" for exam type \"{exam_type}\" redefines an " \
                                    f"existing template with different data:\n"
                                if existing["reader_view"] != new_reader_view:
                                    error_message += "- mismatch in referenced template definition file\n"
                                if existing["explanation"] != template_explanation:
                                    error_message += "- mismatch in template explanation\n"
                                error_message += "Please revise the index."
                                raise ValueError(error_message)
                        else:
                            self._template_data[template_name] = {
                                "explanation": template_explanation,
                                "model": template_model,
                                "reader_view": new_reader_view,
                                "entry_name": template_entry["name"]
                            }
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            f"Could not find template file specified in entry for modality \"{modality}\" and exam type \"{exam_type}\": \n"
                            f"\"{template_entry}\"\n Please revise the entry.")
                    except (AttributeError, IndexError, yaml.scanner.ScannerError) as e:
                        raise ValueError(f"Error for template at \"{template_filepath}\". {e}")
                    if template_name not in self._template_tree[modality][exam_type]:
                        self._template_tree[modality][exam_type].append(template_name)
                    else:
                        raise ValueError(f"Template \"{template_entry["name"]}\" with multiple template index entries. Please revise the index.")
        # make sure all template references have been resolved
        for template_name in unresolved_parent_template_references.keys():
            if template_name not in self._template_data.keys():
                raise ValueError(f"Provided template index lists a child template \"{template_name}\" at "
                                 f"\"{unresolved_parent_template_references[template_name]}\" which is not explicitly referenced by any parent "
                                 f"template entry occurring before it.")
        # add combined exams
        self._exist_combined_exam_classes = True if combined_exams_path is not None else False
        if self._exist_combined_exam_classes:
            # load exam type definitions
            combined_exam_defs = yaml.safe_load(open(combined_exams_path))
            # number of templates, that can be used for structuring for each combined exam type
            self._max_templates = dict()
            for modality in combined_exam_defs.keys():
                for exam_defs in combined_exam_defs[modality]:
                    exam_type = exam_defs["Type"]
                    base_types = exam_defs["Definition"]["combines"]  # base exam types combined in exam type
                    # make sure all base types are defined in template index
                    for base_type in base_types:
                        if base_type not in self._template_tree[modality].keys():
                            raise ValueError(f"Undefined base exam type \"{base_type}\" used in definition for combined exam \"{exam_type}\".")
                    # number of base exams that make up combination -> max number of templates for structuring
                    self._max_templates[exam_type] = len(base_types)
                    # all templates of any base exam type in the combination can be selected
                    template_names = [template_name for base_type in base_types for template_name in self._template_tree[modality][base_type]]
                    # remove duplicates preserving order
                    template_names = list(OrderedDict.fromkeys(template_names).keys())
                    # add entry to template tree
                    self._template_tree[modality][exam_type] = template_names
        self._api = api
        if api == "ollama":
            self._llm = ChatOllama(model=model_name)
        elif api == "openai":
            if base_url is None or api_key is None:
                raise ValueError("Must set both \"api_key\" and \"base_url\" arguments to communicate with the OpenAI-compatible server.")
            self._llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, reasoning_effort=reasoning_effort)
        else:
            raise ValueError(f"Unsupported inference backend \"{api}\". Supported backends are \"ollama\" and \"vllm\".")
        # setup for later report type identification
        exam_types_for_modality = {modality: [exam_type for exam_type in self._template_tree[modality].keys()]
                                   for modality in self._template_tree.keys()}
        self._identifier = ReportTypeIdentifier(model_name, raw_report_type_tree=exam_types_for_modality,
                                                categorize_novelties=False, api=api, base_url=base_url, api_key=api_key,
                                                reasoning_effort=reasoning_effort)
        self._special_responses = SPECIAL_RESPONSES
        self._prompt_templates = PROMPT_TEMPLATES["generator"]
        self._no_appropriate_template_response = self._special_responses["other_response"]
        self._information_missing_response = "nicht erwähnt"
        self._retry_kwargs = dict(retry_if_exception_type=(ValidationError, JSONDecodeError))

    def _structured_llm(self, response_template: Type[BaseModel]) -> Runnable:
        return self._llm.with_structured_output(schema=response_template, method="json_schema").with_retry(**self._retry_kwargs)

    def get_index_entry_name(self, template_name: str) -> str:
        """
        :param template_name: name of pydantic model representing template
        :return: name of entry in template index file
        """
        return self._template_data[template_name]["entry_name"]

    def get_pipeline(self, prompt_template_key: str, response_template: Type[BaseModel]) -> Runnable:
        """
        create structured output pipeline with retries using attributes
        :param prompt_template_key:
        :param response_template:
        :return:
        """
        return self._prompt_templates[prompt_template_key] | self._structured_llm(response_template)

    def select_templates(self, report_text: str) -> dict[str: str | list[str]]:
        """
        selects the appropriate structuring template for the provided radiology report of unknown type by
        1. identifying its examination modality & type (body region)
        2. selecting an appropriate structuring template based on the provided explanations

        :param report_text: radiology report to select template for
        :return: dictionary containing:
            - "modality": identified examination modality, or a sentinel value from utils.SPECIAL_RESPONSES
              ("no_findings_response", "other_response") if the report could not be classified
            - "exam_types": identified examination types, or a single-element list containing a sentinel
              value from SPECIAL_RESPONSES if the exam type could not be identified
            - "template_names": names of selected structuring templates; falls back to a single-element list
              containing the "default_field_value" sentinel ("") if modality/exam type could not be
              identified or the LLM failed to return a valid template selection, or the "other_response"
              sentinel if the LLM determined none of the offered templates fit the report
        """
        # identify modality & exam type
        modality, exam_type = self._identifier.identify_report_type(report_text, extract_main_diagnosis=False)
        # catch special responses
        if modality in self._special_responses.values() or exam_type in self._special_responses.values():
            if modality not in self._special_responses.values() and exam_type == self._special_responses["multiple_exams_response"]:
                # LLM indicates multiple exams detailed
                return self.select_templates_for_multiple_exams_detailed(report_text)
            return {
                "modality": modality,
                "exam_types": [exam_type, ],
                "template_names": [self._special_responses["default_field_value"], ]
            }
        else:
            # retrieve template names for exam type(s)
            template_names = [template_name for template_name in self._template_tree[modality][exam_type]]
            # remove duplicates preserving order
            template_names = list(OrderedDict.fromkeys(template_names).keys())
            # create reader view with template explanations for LLM
            options_reader_view = "\n".join([f"\"{template_name}\": \"{self._template_data[template_name]["explanation"]}\""
                                             for template_name in template_names])
            # select template based on explanations
            selection_options = template_names + [self._no_appropriate_template_response]
            # determine whether single template sufficient to structure report (base exam)
            is_base_exam = True if not self._exist_combined_exam_classes or exam_type not in self._max_templates.keys() else False
            if is_base_exam:
                # Using eval() to create Literal types dynamically from trusted internal values
                response_template = create_model("AusgewaehltesTemplateAngabe",
                                                 template_name=(eval(f"Literal{selection_options}"), self._special_responses["default_field_value"]))
                # single choice selection pipeline
                selection_pipeline = self.get_pipeline("select_template", response_template)
            else:
                # Using eval() to create Literal types dynamically from trusted internal values
                response_template = create_model("AusgewaehlteTemplatesAngabe",
                                                 template_names=(
                                                     eval(f"list[Literal{selection_options}]"), [self._special_responses["default_field_value"]]))
                # multiple choice selection pipeline
                selection_pipeline = self.get_pipeline("select_template, multi_choice", response_template)
            try:
                pipeline_args = {
                    "template_names_and_explanations": options_reader_view,
                    "response_template": response_template().model_dump_json(),
                    "report_text": report_text,
                    **self._special_responses
                }
                if not is_base_exam:
                    pipeline_args["n_max_templates"] = self._max_templates[exam_type]
                response_model_instance = selection_pipeline.invoke(pipeline_args)
                selected_template_names = [response_model_instance.model_dump()["template_name"], ] if is_base_exam \
                    else response_model_instance.model_dump()["template_names"]
            except (ValidationError, JSONDecodeError, OutputParserException):  # no valid response
                warn(f"LLM failed to provide an appropriate template selection response within the allowed number of tries.")
                return {
                    "modality": modality,
                    "exam_types": [exam_type, ],
                    "template_names": [self._special_responses["default_field_value"], ]
                }
            if selected_template_names in self._special_responses.values():
                return {
                    "modality": modality,
                    "exam_types": [exam_type, ],
                    "template_names": selected_template_names
                }
        return {
            "modality": modality,
            "exam_types": [exam_type, ],
            "template_names": selected_template_names
        }

    def select_templates_for_multiple_exams_detailed(self, report_text: str) -> dict[str: str | list[str]]:
        """
        Experimental implementation of template selection for reports that detail results of multiple examinations.

        :param report_text: radiology report text detailing results of multiple examinations
        :return: dictionary containing:
            - "exam_types": identified examination types
            - "template_names": names of selected structuring templates
        """
        # select exam types
        modality, exam_types = self._identifier.identify_report_type_multi_choice(report_text)
        # catch special responses
        if modality in self._special_responses.values() or (
                len(exam_types) == 1 and exam_types[0] in self._special_responses.values()):  # LLM indicates special case
            return {
                "modality": modality,
                "exam_types": exam_types,
                "template_names": [self._special_responses["default_field_value"], ]
            }
        # retrieve template names for exam types
        template_names = [template_name for exam_type in exam_types if exam_type not in self._special_responses.values()  # skip special responses
                          for template_name in self._template_tree[modality][exam_type]]
        # remove duplicates preserving order
        template_names = list(OrderedDict.fromkeys(template_names).keys())
        # create reader view with template explanations for LLM
        options_reader_view = "\n".join([f"\"{template_name}\": \"{self._template_data[template_name]["explanation"]}\""
                                         for template_name in template_names])
        # select template based on explanations
        selection_options = template_names + [self._no_appropriate_template_response]
        # create selection pipeline
        response_template = create_model("AusgewaehltesTemplateAngabe",
                                         template_names=(
                                             eval(f"list[Literal{selection_options}]"), [self._special_responses["default_field_value"]]))
        # multiple choice selection pipeline
        selection_pipeline = self.get_pipeline("select_template, multi_choice, multiple_exams", response_template)
        pipeline_args = {
            "template_names_and_explanations": options_reader_view,
            "response_template": response_template().model_dump_json(),
            "report_text": report_text,
            **self._special_responses
        }
        try:
            response_model_instance = selection_pipeline.invoke(pipeline_args)
            selected_template_names = response_model_instance.model_dump()["template_names"]
        except (ValidationError, JSONDecodeError, OutputParserException):
            warn(f"LLM failed to provide an appropriate template selection response within the allowed number of tries.")
            return {
                "modality": modality,
                "exam_types": exam_types,
                "template_names": [self._special_responses["default_field_value"]]
            }
        return {
            "modality": modality,
            "exam_types": exam_types,
            "template_names": selected_template_names
        }

    def structure_report(self, report_text: str, template_names: list[str]) -> str | list[str]:
        """
        structures radiology report in accordance with the specified structuring template(s)
        :param report_text: radiology report to structure
        :param template_names: name(s) of structuring template(s) to use
        :return: structured report(s) as (list of) JSON-formatted strings
        :raises ValueError: if template_name is unknown or missing metadata
        :raises ValueError: if LLM fails to provide appropriate(ly structured) response after 2 retries
        """
        if len(template_names) == 1:
            template_name = template_names[0]
        else:
            return self.structure_report_sequentially(report_text, template_names)
        if template_name == self._special_responses["other_response"] or template_name == self._special_responses["default_field_value"]:
            raise ValueError("No appropriate template found or no template provided.")
        elif template_name not in self._template_data.keys():
            raise ValueError(f"Unknown template \"{template_name}\"")
        structuring_pipeline = self.get_pipeline("structure_report", self._template_data[template_name]["model"])
        try:
            response_model_instance = structuring_pipeline.invoke({
                "template_reader_view": self._template_data[template_name]["reader_view"],
                "information_missing_response": self._information_missing_response,
                "report_text": report_text
            })
            structured_report = response_model_instance.model_dump_json()
        except (ValidationError, JSONDecodeError, OutputParserException):  # no valid response
            raise ValueError(f"LLM failed to structure report using template \"{template_name}\".")
        return structured_report

    def structure_report_sequentially(self, report_text: str, template_names: list[str]) -> list[str]:
        """
        experimental implementation of structuring a report detailing a combined examination / the results of multiple examinations by having the LLM
        fill in multiple templates sequentially
        :param report_text: radiology report to structure
        :param template_names: name of structuring template to use
        :return: list of structured reports as formatted JSON strings
        :raises ValueError: if template_names contains a sentinel value or a name unknown to the template index
        :raises ValueError: if LLM fails to provide appropriate(ly structured) response after 2 retries
        """
        if any([tn in self._special_responses.values() for tn in template_names]):
            raise ValueError("No appropriate template found or no template provided for one of the detailed examinations.")
        elif any([tn not in self._template_data.keys() for tn in template_names]):
            raise ValueError(f"Unknown selected template for one of the detailed examinations")
        # collect template name and template reader view for provided template names
        template_info = []
        for template_name in template_names:
            template_info.append(f"Template-Titel: \n {template_name}")
            template_info.append(f"Template: \n {self._template_data[template_name]["reader_view"]}")
        # join to single string
        template_info = "\n".join(template_info)
        # start message history
        messages = self._prompt_templates["structure_report, sequentially"]["initial"].format_messages(**{
            "template_info": template_info,
            "information_missing_response": self._information_missing_response,
            "report_text": report_text,
            "template_title": template_names[0]
        })
        structured_reports = []  # structured report strings
        first_turn = True  # first turn of conversation
        for template_name in template_names:
            # setup llm to produce correct output
            response_template = self._template_data[template_name]["model"]
            llm = self._structured_llm(response_template)
            if not first_turn:
                # add follow-up prompt
                messages.append(
                    self._prompt_templates["structure_report, sequentially"]["follow-up"].format_messages(template_title=template_name)[0])
            # get response
            try:
                response_model_instance = llm.invoke(messages)
            except (ValidationError, JSONDecodeError, OutputParserException):  # no valid response
                raise ValueError(f"LLM failed to structure report using template \"{template_name}\".")
            # save response
            structured_reports.append(response_model_instance.model_dump_json())
            # add response to history
            messages.append(AIMessage(content=structured_reports[-1]))
            first_turn = False
        return structured_reports
