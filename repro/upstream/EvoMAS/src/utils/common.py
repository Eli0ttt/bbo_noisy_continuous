from pydantic import BaseModel
import yaml
import logging
import pandas as pd
import re
import asyncio
from typing import Optional, Any, List, Coroutine, Dict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def read_yaml_file(file_path: str) -> dict:
    """
    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A Python dictionary representing the YAML content,
              return {} if the file is not found or an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            yaml_content = yaml.safe_load(file)
            if yaml_content is None:
                return {}
            else:
                return dict(yaml_content)
    except FileNotFoundError:
        logger.error(f"Error: File not found at '{file_path}'")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{file_path}': {e}")
        return {}


def read_xlsx(file_path: str) -> Optional[pd.DataFrame]:
    """Read xlsx file from specified file_path.

    Args:
        file_path: Path of the Excel file.

    Returns:
        pd.DataFrame: Pandas Dataframe extracted from the file if
        successful
    """
    logger.info(f"File {file_path} reading...")
    try:
        df = pd.read_excel(file_path)
        logger.info(f"File {file_path} read successfully.")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise e
    return df


def write_xlsx(file_path: str, data: pd.DataFrame, sheet_name: str = "Sheet1", index: bool = False) -> None:
    """Write xlsx file to specified file_path.

    Args:
        file_path: Path of the Excel file.
        data: Pandas Dataframe
        sheet_name: Sheet name of excel file to write to
        index: Whether to write the index

    """
    logger.info(f"File {file_path} saving...")
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=index)
            logger.info(f"DataFrame appended to '{file_path}' sheet '{sheet_name}'")
    except FileNotFoundError:
        # If the file doesn't exist, create it with the first DataFrame
        data.to_excel(file_path, sheet_name=sheet_name, index=index, engine="openpyxl")
        logger.info(f"'{file_path}' created with sheet '{sheet_name}'.")
    except ImportError:
        logger.error("Error: You need to install the 'openpyxl' library to work with Excel files.")


def extract_content_from_tag(content: str, tag: str) -> Optional[str]:
    """
    Extracts content from a given XML/HTML-like tag in a string using regex.
    This is robust against malformed structures that can cause XML parsers to fail.

    Args:
        content: The string containing the tag.
        tag: The name of the tag to extract content from (e.g., "planning").

    Returns:
        The extracted content as a string, or None if not found.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def remove_outer_tags_if_any(text: str) -> str:
    """
    Extracts content from a given XML/HTML-like outer tag in a string using regex.

    Args:
        text: The string containing the tag or not.

    Returns:
        The extracted content as a string, or the original text if not found.
    """
    pattern = r"^\s*<([^>/\s]+)[^>]*>(.*)</\1>\s*$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text


# Helper function to run async code in a thread
def run_async_in_thread(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Runs an async coroutine in a dedicated asyncio event loop within the current thread.
    Each thread created by ThreadPoolExecutor will call this, creating its own loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # No current event loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_or_create_event_loop() -> Any:
    try:
        # If we're already in a running loop (inside async context)
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            # In older asyncio contexts (e.g. Python <3.10)
            return asyncio.get_event_loop()
        except RuntimeError:
            # In a thread where no loop is set: create and set one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def check_required_fields_in_data_model(data_model: BaseModel, required_fields: List[str]) -> bool:
    """
    Check if required fields exist and not None in a Pydantic BaseModel.

    Args:
        content: The data model.
        required_fields: The name of the required fields.

    Returns:
        True if all required fields exist and not None, False otherwise.
    """
    try:
        for field in required_fields:
            if getattr(data_model, field) is None:
                logger.error(f"Error getting {field} from data_model {data_model}")
                return False
    except Exception as e:
        logger.error(f"Error checking required fields in data_model {data_model}: {e}")
        return False
    return True


def ensemble_prompt(prompt_template: str, data_model: BaseModel) -> str:
    """Ensemble prompt from template and data model.

    Args:
        prompt_template: Template of the prompt.
        data_model: Data model containing variables and values.

    Returns:
        str: Ensembled prompt filled with data model variables and values
    """
    mappings = data_model.model_dump()
    for key in list(mappings.keys()):
        new_key = f"{{{{{key}}}}}"
        mappings.update({new_key: mappings.pop(key)})
    for key, value in mappings.items():
        if value is None:
            value = ""
        elif isinstance(value, str):
            value = value.strip()
        elif isinstance(value, int):
            value = str(value).strip()
        elif isinstance(value, list) or isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False).strip()
        else:
            raise TypeError("Unsupported type {} for value {}".format(type(value), value))
        prompt_template = prompt_template.replace(key, value)
    return prompt_template


def extract_variables_from_sp(text: str) -> List[Any]:
    """Extract user prompts variables in system prompt.

    Args:
        text: System prompt containing user prompts variables.

    Returns:
        List[str]: user prompts variables extracted from the system prompt
    """
    # Regular expression pattern to match variables within double brackets
    pattern = r"\{\{([^}]+)\}\}"
    # Find all matches in the text
    matches = re.findall(pattern, text)
    # Return the list of extracted variables
    return matches


def parse_paths_from_response(response: Any) -> List[str]:
    """Extract file paths from response (agent output, string, etc.)"""
    text = str(response.content if hasattr(response, 'content') else response)
    paths = []
    
    # Try JSON array first
    try:
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            parsed = json.loads(text[start:end])
            if isinstance(parsed, list):
                paths = [item for item in parsed if isinstance(item, str) and '.json' in item]
                if paths:
                    return paths
    except:
        pass
    
    # Fallback: regex extraction
    matches = re.findall(r'[A-Za-z0-9_\-/]+\.json', text)
    seen = set()
    for match in matches:
        if match not in seen:
            paths.append(match)
            seen.add(match)
    
    return paths


def read_workflow_configs(file_paths: List[str], base_dir: str = "dataset/configurations") -> List[Dict[str, Any]]:
    """Read workflow JSONs and return configs with text format"""
    workflows = []
    base_path = Path(base_dir)
    
    for file_path in file_paths:
        full_path = base_path / file_path
        if not full_path.exists():
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            workflows.append({
                "path": file_path,
                "config": config,
                "text": json.dumps(config, indent=2, ensure_ascii=False)
            })
        except:
            continue
    
    return workflows


def format_workflows_as_fewshots(workflows: List[Dict[str, Any]], max_length: int = None) -> str:
    """Format workflows as few-shot examples"""
    if not workflows:
        return ""
    
    examples = []
    for i, workflow in enumerate(workflows, 1):
        config_text = workflow['text']
        if max_length and len(config_text) > max_length:
            config_text = config_text[:max_length] + "\n... (truncated)"
        
        examples.append(f"Example {i}: {workflow['path']}\n```json\n{config_text}\n```")
    
    return "\n\n".join(examples)


if __name__ == "__main__":

    class TestModel(BaseModel):
        a: Optional[str] = "a"
        b: Optional[str] = "b"

    check_required_fields_in_data_model(TestModel(), ["a", "b"])
    check_required_fields_in_data_model(TestModel(), ["a", "c"])
    check_required_fields_in_data_model(TestModel(a=None), ["a", "b"])
    check_required_fields_in_data_model(TestModel(b=None), ["a", "c"])
    check_required_fields_in_data_model(TestModel(a=None, b=None), ["a", "b"])
