from pydantic import create_model, BaseModel
from typing import Any, Type, Tuple
from copy import deepcopy
import re

NON_ASCII_CHARS_TRANSCRIPTION = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}

def to_class_name(preliminary_name: str) -> str:
    """
    Convert a preliminary name to a valid Python class name in UpperCamelCase.

    :param preliminary_name: The preliminary name to convert
    :return: Valid Python class name
    """
    # convert attribute name to upper camel case
    class_name = "".join([word.capitalize() for word in preliminary_name.split(" ")])
    # replace non ascii characters
    class_name = "".join([char if ord(char) < 128 else NON_ASCII_CHARS_TRANSCRIPTION[char]
                          for char in class_name])
    # replace special chars with underscore
    special_char = r'[\(\)\[\]\-/,]'
    class_name = "".join(['_' if re.match(special_char, char) else char for char in class_name])
    return class_name

def to_template_name(template_title: str) -> str:
    """
    Convert a template index entry title to a template class name.

    :param template_title: The template index entry title to convert
    :return: Template class name
    """
    return to_class_name(template_title) + "Template"

def class_name_to_attribute_name(class_name: str) -> str:
    """
    Convert a class name to an attribute name in lowerCamelCase.

    :param class_name: The class name to convert
    :return: Attribute name in lowerCamelCase
    """
    # convert class name to lowerCamelCase
    attribute_name = class_name[0].lower() + class_name[1:]
    return attribute_name

def non_composite_to_string(field_instructions: dict[str, Any] | list | Any) -> dict[str, Any] | list | str:
    """
    Recursively converts all field instructions that do not define composite types (list or pydantic model) to string.

    :param field_instructions: Field instructions to convert (can be dict, list, or immutable type)
    :return: Converted field instructions with non-composite types as strings
    """
    if isinstance(field_instructions, dict):
        for key in field_instructions.keys():
            field_instructions[key] = non_composite_to_string(field_instructions[key])
    elif isinstance(field_instructions, list):
        if not field_instructions:
            updated_list = [""]
        else:
            updated_list = []
            for item in field_instructions:
                item = non_composite_to_string(item)
                updated_list.append(item)
        field_instructions = updated_list
    elif field_instructions is None:
        field_instructions = ""
    else:
        field_instructions = str(field_instructions)
    return field_instructions

def instructions_to_pydantic_model(model_instructions: dict[str, Any], model_name: str) \
        -> Tuple[Type[BaseModel] | Type[str] | Type[float] | Type[list], BaseModel | str | float | list[BaseModel | str]]:
    """
    Creates a pydantic model based on the instructions laid out in model_instructions.

    :param model_instructions: Dictionary with keys defining field names and values defining field types:
        - str: field type is String
        - dict[str, Any]: field type is a nested pydantic model
        - list[str | dict[str, Any]]: field type is list[str] or list[BaseModel]
    :param model_name: Name of the pydantic model class to return
    :return: Tuple of (pydantic model type, default value for model instance)
    """
    if isinstance(model_instructions, str):
        # string field (base case)
        return str, ""
    elif isinstance(model_instructions, list):
        element_type, default_value = instructions_to_pydantic_model(model_instructions[0], model_name)
        return list[element_type], [default_value, ]
    elif isinstance(model_instructions, dict):
        # generate kwargs for create_model
        pydantic_model_dict = {}
        for key in model_instructions.keys():
            # determine field type
            field_type, default_value = instructions_to_pydantic_model(model_instructions[key], key[0].upper() + key[1:])
            # add entry for create_model
            pydantic_model_dict[key] = (field_type, default_value)
        pydantic_model = create_model(model_name, **pydantic_model_dict)
        return pydantic_model, pydantic_model()
    else:
        raise TypeError(f"Invalid instructions for field \"{model_name}\" (instruction type \"{type(model_instructions)}\"). "
                        f"Supported instruction types are list, dict and str.")

def extend_instructions(base: dict[str, Any] | str, extension: dict[str, Any] | str) -> dict[str, Any] | list[dict[str, Any] | str] | str:
    """
    Extends the type instructions provided by base as specified by extension by updating extension recursively.

    :param base: default value string or dictionary with instructions for creating a pydantic model (base model)
    :param extension: default value string or dictionary with instructions for additional / updated fields for base model
    :return: updated instructions containing all additional / updated field instructions as specified by extension
    :raises TypeError: If there's a mismatch in field type definitions between parent and child templates
    """
    if isinstance(base, str):
        # either duplicate definitions -> use default value as defined in extension
        # or extension redefines base field with complex type (list or pydantic model) -> overwrite base
        return extension
    else:  # both base & extension define field of complex type (list or pydantic model)
        if type(base) is not type(extension):
            raise TypeError(f"Expected section content definitions of parent and child template files are incompatible (value types are "
                            f"\"{type(base)}\" and \"{type(extension)}\"). The value in the parent file is:\n{base}\nThe incompatible value "
                            f"in the child file is:\n{extension}")
        elif isinstance(base, list):
            return [extend_instructions(base[0], extension[0])]
        else:  # instructions for pydantic model
            result = deepcopy(extension)
            for base_key, base_val in base.items():
                if base_key in result.keys():
                    # both base & extension define field -> update instructions for base field type
                    result[base_key] = extend_instructions(base_val, result[base_key])
                else:
                    # only base defines field -> add field with base definition
                    result[base_key] = base_val
    return result
