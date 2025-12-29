"""
Data loading utilities

Load training data from various formats: JSONL, CSV, DataFrame, HF Dataset
"""

import json
from typing import List, Dict, Union, Any
from pathlib import Path


class DataLoader:
    """Load training data from various sources"""

    def __init__(self):
        """Initialize data loader"""
        pass

    def load(
        self,
        source: Union[str, List[Dict], Any],
        format: str = "auto"
    ) -> List[Dict[str, str]]:
        """
        Load data from various sources.

        Args:
            source: File path, list of dicts, DataFrame, or HF Dataset
            format: "auto", "jsonl", "csv", "dataframe", or "hf_dataset"

        Returns:
            List of {"query": str, "tool": str} dicts

        Examples:
            >>> loader = DataLoader()
            >>> data = loader.load("train.jsonl")
            >>> data = loader.load("train.csv", format="csv")
            >>> data = loader.load(df, format="dataframe")
        """
        # If already a list of dicts, validate and return
        if isinstance(source, list):
            return self._validate_data(source)

        # Detect format from file extension if auto
        if format == "auto" and isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.suffix == ".jsonl":
                format = "jsonl"
            elif source_path.suffix == ".csv":
                format = "csv"
            elif source_path.suffix == ".json":
                format = "json"

        # Load based on format
        if format == "jsonl":
            return self.load_jsonl(source)
        elif format == "json":
            return self.load_json(source)
        elif format == "csv":
            return self.load_csv(source)
        elif format == "dataframe":
            return self.load_dataframe(source)
        elif format == "hf_dataset":
            return self.load_hf_dataset(source)
        else:
            # Try to detect type
            if hasattr(source, "to_dict"):  # pandas DataFrame
                return self.load_dataframe(source)
            elif hasattr(source, "__iter__") and hasattr(source, "features"):  # HF Dataset
                return self.load_hf_dataset(source)
            else:
                raise ValueError(f"Unknown data format: {format}")

    def load_jsonl(self, filepath: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from JSONL file.

        Args:
            filepath: Path to JSONL file

        Returns:
            List of {"query": str, "tool": str} dicts
        """
        data = []
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {filepath}: {e}"
                    )

        return self._validate_data(data)

    def load_json(self, filepath: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from JSON file.

        Args:
            filepath: Path to JSON file (should contain array of objects)

        Returns:
            List of {"query": str, "tool": str} dicts
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")

        return self._validate_data(data)

    def load_csv(self, filepath: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load data from CSV file.

        Expects columns: query, tool

        Args:
            filepath: Path to CSV file

        Returns:
            List of {"query": str, "tool": str} dicts
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for CSV loading. "
                "Install with: pip install pandas"
            )

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        return self.load_dataframe(df)

    def load_dataframe(self, df: Any) -> List[Dict[str, str]]:
        """
        Load data from pandas DataFrame.

        Args:
            df: pandas DataFrame with 'query' and 'tool' columns

        Returns:
            List of {"query": str, "tool": str} dicts
        """
        # Check required columns
        if "query" not in df.columns or "tool" not in df.columns:
            raise ValueError(
                "DataFrame must have 'query' and 'tool' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Convert to list of dicts
        data = df[["query", "tool"]].to_dict("records")

        return self._validate_data(data)

    def load_hf_dataset(self, dataset: Any) -> List[Dict[str, str]]:
        """
        Load data from Hugging Face Dataset.

        Args:
            dataset: HuggingFace Dataset object

        Returns:
            List of {"query": str, "tool": str} dicts
        """
        # Check required columns/features
        if "query" not in dataset.features or "tool" not in dataset.features:
            raise ValueError(
                "Dataset must have 'query' and 'tool' features. "
                f"Found features: {list(dataset.features.keys())}"
            )

        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "query": item["query"],
                "tool": item["tool"]
            })

        return self._validate_data(data)

    def _validate_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate that data has correct format.

        Args:
            data: List of dicts to validate

        Returns:
            Validated data

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if len(data) == 0:
            raise ValueError("Data is empty")

        # Check each item
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary: {type(item)}")

            if "query" not in item:
                raise ValueError(f"Item {i} missing 'query' field")

            if "tool" not in item:
                raise ValueError(f"Item {i} missing 'tool' field")

            if not isinstance(item["query"], str):
                raise ValueError(
                    f"Item {i} 'query' must be string, got {type(item['query'])}"
                )

            if not isinstance(item["tool"], str):
                raise ValueError(
                    f"Item {i} 'tool' must be string, got {type(item['tool'])}"
                )

        return data

    def save_jsonl(self, data: List[Dict[str, str]], filepath: Union[str, Path]):
        """
        Save data to JSONL file.

        Args:
            data: List of {"query": str, "tool": str} dicts
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    def save_json(self, data: List[Dict[str, str]], filepath: Union[str, Path]):
        """
        Save data to JSON file.

        Args:
            data: List of {"query": str, "tool": str} dicts
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
