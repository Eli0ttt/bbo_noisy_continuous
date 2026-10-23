import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TaskData(BaseModel):
    id: Union[int, str]  # Allow both integer and string IDs for compatibility
    query: str
    gt: str
    tag: List[str]
    source: str
    metadata: Optional[Dict[str, Any]] = None  # For SWE-bench and other datasets with extra metadata


class Dataset:
    """
    Dataset class to handle loading and accessing task data from various dataset structures.

    Supports:
    - BBEH benchmark_tasks subdirectories (e.g., bbeh_word_sorting, bbeh_nycc)
    - Direct dataset files (e.g., gsm8k/test.json, math/train.json)
    """

    def __init__(self, dataset_name: str, split: str = "task"):
        """
        Initialize dataset loader.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.dataset_root = os.path.join(self.repo_root, "dataset")

        self.data: List[TaskData] = []
        self.dataset_path: Optional[str] = None
        self._load_dataset()

    def _find_dataset_path(self) -> str:
        """
        Find the appropriate dataset path based on dataset name and structure.
        """
        # Handle BBEH benchmark tasks
        if self.dataset_name.startswith('bbeh_'):
            bbeh_path = os.path.join(self.dataset_root, "bbeh", "benchmark_tasks", self.dataset_name, f"{self.split}.json")
            if os.path.exists(bbeh_path):
                return bbeh_path

        # Handle SWE-bench datasets
        if self.dataset_name.startswith('swe') or self.dataset_name in ['swebench', 'swebench_lite', 'swebench_verified']:
            # Map common SWE-bench name variations to directory names
            swe_bench_mappings = {
                'swebench': 'swe_bench',
                'swebench_lite': 'swe_bench_lite',
                'swebench_verified': 'swe_bench_verified',
                'swe-bench': 'swe_bench',
                'swe-bench-lite': 'swe_bench_lite',
                'swe-bench-verified': 'swe_bench_verified',
                'swe_bench': 'swe_bench',
                'swe_bench_lite': 'swe_bench_lite',
                'swe_bench_verified': 'swe_bench_verified'
            }

            swe_dir = swe_bench_mappings.get(self.dataset_name, self.dataset_name)
            swe_path = os.path.join(self.dataset_root, swe_dir, f"{self.split}.json")
            if os.path.exists(swe_path):
                return swe_path

        # Handle WorkBench datasets (workbench_analytics, workbench_calendar, etc.)
        if self.dataset_name.startswith('workbench_'):
            domain = self.dataset_name.replace('workbench_', '')
            workbench_path = os.path.join(self.dataset_root, "workbench", domain, f"{self.split}.json")
            if os.path.exists(workbench_path):
                return workbench_path

        # Standard dataset patterns
        possible_patterns = [
            f"{self.dataset_name}/{self.split}.json",
            f"{self.dataset_name}/task.json",
            f"{self.dataset_name}.json",
            f"{self.dataset_name}/{self.dataset_name.upper()}.json"
        ]

        for pattern in possible_patterns:
            full_path = os.path.join(self.dataset_root, pattern)
            if os.path.exists(full_path):
                return full_path

        # Last resort: search for any JSON files in the dataset directory
        dataset_dir = os.path.join(self.dataset_root, self.dataset_name)
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.endswith('.json'):
                    return os.path.join(dataset_dir, file)

        raise FileNotFoundError(f"Dataset '{self.dataset_name}' with split '{self.split}' not found")

    def _load_dataset(self) -> None:
        """Load dataset from the determined path."""
        try:
            self.dataset_path = self._find_dataset_path()

            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Handle different data structures
            if isinstance(raw_data, list):
                # Data is already a list of tasks
                task_list = raw_data
            elif isinstance(raw_data, dict):
                for key in ['tasks', 'data', 'examples', 'test', 'train']:
                    if key in raw_data and isinstance(raw_data[key], list):
                        task_list = raw_data[key]
                        break
                else:
                    task_list = [raw_data] if self._is_single_task(raw_data) else []
            else:
                raise ValueError(f"Unexpected data format in {self.dataset_path}")

            self.data = []
            for i, task in enumerate(task_list):
                try:
                    # Handle missing fields with defaults
                    if 'id' not in task:
                        task['id'] = i
                    if 'tag' not in task:
                        task['tag'] = [self.dataset_name]
                    if 'source' not in task:
                        task['source'] = self.dataset_name.upper()

                    # Ensure required fields exist
                    required_fields = ['query', 'gt']
                    for field in required_fields:
                        if field not in task:
                            # Try alternative field names
                            alternatives = {
                                'query': ['question', 'problem', 'input', 'text'],
                                'gt': ['answer', 'ground_truth', 'target', 'label', 'output']
                            }

                            found = False
                            for alt in alternatives.get(field, []):
                                if alt in task:
                                    task[field] = task[alt]
                                    found = True
                                    break

                            if not found:
                                raise KeyError(f"Required field '{field}' not found in task {i}")

                    # Convert to TaskData
                    validated_task = TaskData(**task)
                    self.data.append(validated_task)

                except Exception as e:
                    logger.warning(f"Skipping invalid task {i} in {self.dataset_name}: {e}")
                    continue

            if not self.data:
                raise ValueError(f"No valid tasks found in dataset {self.dataset_name}")

            logger.info(f"Loaded {len(self.data)} tasks from {self.dataset_path}")

        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {e}")
            raise

    def _is_single_task(self, data: Dict[str, Any]) -> bool:
        """Check if a dictionary represents a single task rather than a collection."""
        # Look for task-like fields
        task_fields = {'query', 'question', 'problem', 'input', 'gt', 'answer', 'target'}
        return any(field in data for field in task_fields)

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> TaskData:
        """Get a task by index."""
        return self.data[index]

    def __iter__(self):
        """Iterate over tasks in the dataset."""
        return iter(self.data)

    def get_by_id(self, task_id: Union[int, str]) -> Optional[TaskData]:
        """
        Get a task by its ID field. Accepts int indices (BBEH/WorkBench) or
        string instance IDs (SWE-bench, e.g. 'astropy__astropy-12907').
        """
        for task in self.data:
            if task.id == task_id:
                return task
        return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_tasks": len(self.data),
            "dataset_path": self.dataset_path,
            "sample_task_ids": [task.id for task in self.data[:5]]  # First 5 task IDs
        }


def load_dataset(dataset_name: str, split: str = "task") -> Dataset:
    return Dataset(dataset_name, split)


def list_available_datasets() -> Dict[str, List[str]]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_root = os.path.join(repo_root, "dataset")

    datasets = {}

    # Find BBEH benchmark tasks
    bbeh_path = os.path.join(dataset_root, "bbeh", "benchmark_tasks")
    if os.path.exists(bbeh_path):
        for task_dir in os.listdir(bbeh_path):
            task_path = os.path.join(bbeh_path, task_dir)
            if os.path.isdir(task_path):
                splits = []
                for file in os.listdir(task_path):
                    if file.endswith('.json'):
                        split_name = file[:-5]
                        splits.append(split_name)

                if splits:
                    datasets[task_dir] = splits

    # Find WorkBench domains
    workbench_path = os.path.join(dataset_root, "workbench")
    if os.path.exists(workbench_path):
        for domain_dir in os.listdir(workbench_path):
            domain_path = os.path.join(workbench_path, domain_dir)
            if os.path.isdir(domain_path):
                splits = []
                for file in os.listdir(domain_path):
                    if file.endswith('.json'):
                        split_name = file[:-5]
                        splits.append(split_name)

                if splits:
                    datasets[f"workbench_{domain_dir}"] = splits

    # Find SWE-bench and other datasets
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path) and item not in ["bbeh", "workbench"]:
            splits = []
            for file in os.listdir(item_path):
                if file.endswith('.json'):
                    split_name = file[:-5]
                    splits.append(split_name)

            if splits:
                # For SWE-bench datasets, also add common alternative names
                datasets[item] = splits
                if item.startswith('swe_bench'):
                    # Add hyphenated versions as well
                    alt_name = item.replace('_', '-')
                    datasets[alt_name] = splits
                    # Add compact versions
                    if item == 'swe_bench':
                        datasets['swebench'] = splits
                    elif item == 'swe_bench_lite':
                        datasets['swebench_lite'] = splits
                    elif item == 'swe_bench_verified':
                        datasets['swebench_verified'] = splits

    return datasets


if __name__ == "__main__":
    print("Available datasets:")
    available = list_available_datasets()
    for dataset_name, splits in available.items():
        print(f"  {dataset_name}: {splits}")
