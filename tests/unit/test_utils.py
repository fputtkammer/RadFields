import os
from typing import Any
import yaml
import pytest

from src.rad_fields.utils import instructions_to_pydantic_model, extend_instructions, non_composite_to_string
from tests.data.mri_prostate_pca import MRIProstatePCaTemplate


def get_field_ids(instructions: dict[str, Any] | list[str | Any] | str, current_id: str = "") -> list[str]:
    """
    Recursively generates a list of IDs for fields that would be expected in pydantic model generated from instructions.

    :param instructions: Field instructions (dict, list, or str)
    :param current_id: Current field ID (used for recursion)
    :return: List of field IDs
    """
    if isinstance(instructions, str):
        return [current_id]
    elif isinstance(instructions, list):
        return get_field_ids(instructions[0])
    elif isinstance(instructions, dict):
        field_key_ids = [current_id]
        for key in instructions.keys():
            field_key_ids += get_field_ids(instructions[key], f"{current_id}/{key}")
        return field_key_ids
    else:
        raise TypeError(f"Instructions for field with id \"{current_id}\" are of invalid type \"{type(instructions).__name__}\".")


def test_instructions_to_pydantic_model() -> None:
    """
    Tests whether pydantic model returned by instructions_to_pydantic_model has same JSON schema representation as
    reference model.
    """
    # reconstruct template instructions from default instance of corresponding pydantic model
    template_dict = MRIProstatePCaTemplate().model_dump()
    template, _ = instructions_to_pydantic_model(template_dict, MRIProstatePCaTemplate.__name__)
    assert template.model_json_schema() == MRIProstatePCaTemplate.model_json_schema()
    # check loaded yaml files are correctly used to create pydantic model
    template_dir = "tests/data/templates/ultrasound"
    template_paths = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(template_dir)
                      for filename in filenames if filename.endswith(".yaml") and not filename.startswith("template_index")]
    files_w_conversion_errors = []
    for template_path in template_paths:
        template_instructions = yaml.safe_load(open(template_path))
        template_instructions = non_composite_to_string(template_instructions)
        try:
            model, _ = instructions_to_pydantic_model(template_instructions, "TestTemplate")
            # check that all fields are included in pydantic model
            assert get_field_ids(template_instructions) == get_field_ids(model().model_dump())
        except AttributeError:
            files_w_conversion_errors.append(template_path)
    assert files_w_conversion_errors == []


def test_extend_instructions() -> None:
    """
    Tests whether extend_instructions produces expected output for representative example dictionaries.
    """
    # test replacement of base type definition with extension type definition
    base = {"field1_1": ""}
    for extension_val in ["new_default_value", [""], [{"field2_1": ""}], {"field2_1": ""}]:
        extension = {list(base.keys())[0]: extension_val}
        assert extend_instructions(base, extension) == extension

    # test extension of pydantic model definitions

    # field of type pydantic model
    base = {"field1_1": {"field2_1": ""}}
    extension = {"field1_1": {"field2_2": ""}}
    expected_merged_instructions = {"field1_1": {"field2_1": "", "field2_2": ""}}
    assert extend_instructions(base, extension) == expected_merged_instructions

    # field of type list[pydantic.BaseModel]
    def convert_defs_to_list_field(instructions):
        return {key: [val] for key, val in instructions.items()}

    base = convert_defs_to_list_field(base)
    extension = convert_defs_to_list_field(extension)
    expected_merged_instructions = convert_defs_to_list_field(expected_merged_instructions)
    assert extend_instructions(base, extension) == expected_merged_instructions
    # test mismatch in field types handled correctly
    extension = {key: val[0] for key, val in extension.items()}
    with pytest.raises(TypeError, match="Expected section content definitions of parent and child template files are incompatible"):
        extend_instructions(base, extension)

    # test extension adds a new top-level key not present in base
    base = {"field1_1": ""}
    extension = {"field1_1": "", "field1_2": "new_field"}
    assert extend_instructions(base, extension) == extension

    # test base keys absent from extension are passed through
    base = {"field1_1": "", "field1_2": "base_only"}
    extension = {"field1_1": "updated"}
    assert extend_instructions(base, extension) == {"field1_1": "updated", "field1_2": "base_only"}

    # test deeply nested merge (3 levels)
    base = {"field1_1": {"field2_1": {"field3_1": ""}}}
    extension = {"field1_1": {"field2_1": {"field3_2": ""}}}
    expected_merged_instructions = {"field1_1": {"field2_1": {"field3_1": "", "field3_2": ""}}}
    assert extend_instructions(base, extension) == expected_merged_instructions

    # test list[str] field (extension overwrites scalar element)
    base = {"field1_1": [""]}
    extension = {"field1_1": ["new_default"]}
    assert extend_instructions(base, extension) == extension


def test_non_composite_to_str() -> None:
    """
    Tests whether non_composite_to_string correctly converts non-string values of leaf nodes in pydantic model instructions to strings.
    """
    for non_string_value in [None, 0.0, 0, False]:
        assert isinstance(non_composite_to_string(non_string_value), str)
        field_name = "field1"
        pydantic_model_instructions = {field_name: non_string_value}
        assert isinstance(non_composite_to_string(pydantic_model_instructions[field_name]), str)
        list_type_instructions = [non_string_value]
        assert isinstance(non_composite_to_string(list_type_instructions)[0], str)
        list_type_instructions = [pydantic_model_instructions]
        assert isinstance(non_composite_to_string(list_type_instructions)[0][field_name], str)