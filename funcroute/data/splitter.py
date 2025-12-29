"""
Pattern group splitting for preventing data leakage

Based on train.py lines 633-707
"""

import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class PatternGroupSplitter:
    """
    Split training data by pattern groups to prevent data leakage.

    This ensures that variations of the same base pattern stay together
    in the same split (train/val/test), preventing the model from
    memorizing patterns rather than learning to route.

    Based on train.py lines 633-673.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def split(
        self,
        data: List[Dict[str, str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        verify_no_leakage: bool = True,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/val/test by pattern groups.

        Args:
            data: List of {"query": str, "tool": str, "base_pattern": str (optional)}
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
            verify_no_leakage: Whether to verify no query overlap

        Returns:
            (train_data, val_data, test_data)

        Example:
            >>> data = [
            ...     {"query": "Where is my order?", "tool": "manage_order"},
            ...     {"query": "wheres my order", "tool": "manage_order"},
            ... ]
            >>> splitter = PatternGroupSplitter()
            >>> train, val, test = splitter.split(data)
        """
        # Validate data
        if len(data) == 0:
            raise ValueError("Data is empty")

        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        # Group data by base_pattern (or by query if no base_pattern)
        pattern_groups = self._create_pattern_groups(data)

        # Check if we have enough groups for splitting
        if len(pattern_groups) < 3:
            print(f"⚠️  WARNING: Only {len(pattern_groups)} pattern groups - some splits may be small or empty")


        print(f"\nPattern Group Splitting:")
        print(f"  Total groups: {len(pattern_groups)}")
        print(f"  Total samples: {sum(len(g['variations']) for g in pattern_groups)}")

        # Split groups (not individual samples!)
        from sklearn.model_selection import train_test_split

        # First split: train vs (val+test)
        train_groups, temp_groups = train_test_split(
            pattern_groups,
            test_size=(val_ratio + test_ratio),
            random_state=self.seed,
            shuffle=True,
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=(1 - val_size),
            random_state=self.seed,
            shuffle=True,
        )

        print(f"  Train groups: {len(train_groups)}")
        print(f"  Val groups: {len(val_groups)}")
        print(f"  Test groups: {len(test_groups)}")

        # Expand groups to individual samples
        train_samples = self._expand_groups(train_groups)
        val_samples = self._expand_groups(val_groups)
        test_samples = self._expand_groups(test_groups)

        print(f"\nExpanded to samples:")
        print(f"  Train: {len(train_samples)}")
        print(f"  Val: {len(val_samples)}")
        print(f"  Test: {len(test_samples)}")

        # Verify no leakage
        if verify_no_leakage:
            self._verify_no_leakage(train_samples, val_samples, test_samples)

        return train_samples, val_samples, test_samples

    def _create_pattern_groups(
        self, data: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Group data by base_pattern or query.

        Args:
            data: List of samples with query, tool, and optional base_pattern

        Returns:
            List of pattern groups
        """
        # Group by (base_pattern, tool) or (query, tool)
        groups_dict = defaultdict(lambda: {"variations": [], "tool": None})

        for sample in data:
            query = sample["query"]
            tool = sample["tool"]
            base_pattern = sample.get("base_pattern", query)

            # Use (base_pattern, tool) as key
            key = (base_pattern, tool)

            if groups_dict[key]["tool"] is None:
                groups_dict[key]["tool"] = tool
                groups_dict[key]["base_pattern"] = base_pattern

            groups_dict[key]["variations"].append(query)

        # Convert to list
        pattern_groups = []
        for (base_pattern, tool), group_data in groups_dict.items():
            pattern_groups.append({
                "base_pattern": base_pattern,
                "tool": tool,
                "variations": group_data["variations"],
            })

        return pattern_groups

    def _expand_groups(self, groups: List[Dict]) -> List[Dict[str, str]]:
        """
        Expand pattern groups to individual samples.

        Args:
            groups: List of pattern groups

        Returns:
            List of individual samples
        """
        samples = []
        for group in groups:
            for variation in group["variations"]:
                samples.append({
                    "query": variation,
                    "tool": group["tool"],
                    "base_pattern": group["base_pattern"],
                })
        return samples

    def _verify_no_leakage(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
    ):
        """
        Verify no query overlap between splits.

        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples

        Raises:
            ValueError: If overlap detected
        """
        train_queries = set(s["query"] for s in train_samples)
        val_queries = set(s["query"] for s in val_samples)
        test_queries = set(s["query"] for s in test_samples)

        overlap_train_val = train_queries & val_queries
        overlap_train_test = train_queries & test_queries
        overlap_val_test = val_queries & test_queries

        print("\n" + "=" * 80)
        print("DATA LEAKAGE CHECK")
        print("=" * 80)
        print(f"Train-Val overlap: {len(overlap_train_val)} queries")
        print(f"Train-Test overlap: {len(overlap_train_test)} queries")
        print(f"Val-Test overlap: {len(overlap_val_test)} queries")

        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("❌ DATA LEAKAGE DETECTED!")
            if overlap_train_val:
                print(f"\nTrain-Val overlap examples:")
                for q in list(overlap_train_val)[:5]:
                    print(f"  - {q}")
            if overlap_train_test:
                print(f"\nTrain-Test overlap examples:")
                for q in list(overlap_train_test)[:5]:
                    print(f"  - {q}")
            print("=" * 80)
            raise ValueError("Data leakage detected between splits!")
        else:
            print("✅ NO DATA LEAKAGE - Splits are clean!")
            print("=" * 80)

    def split_simple(
        self,
        data: List[Dict[str, str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Simple random split without pattern grouping.

        WARNING: This may cause data leakage. Use split() with pattern
        groups for proper anti-leakage splitting.

        Args:
            data: List of samples
            train_ratio: Ratio for training
            val_ratio: Ratio for validation
            test_ratio: Ratio for test

        Returns:
            (train_data, val_data, test_data)
        """
        from sklearn.model_selection import train_test_split

        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        # First split
        train, temp = train_test_split(
            data,
            test_size=(val_ratio + test_ratio),
            random_state=self.seed,
            shuffle=True,
        )

        # Second split
        val_size = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            temp,
            test_size=(1 - val_size),
            random_state=self.seed,
            shuffle=True,
        )

        print(f"\nSimple split (no pattern grouping):")
        print(f"  Train: {len(train)} samples")
        print(f"  Val: {len(val)} samples")
        print(f"  Test: {len(test)} samples")
        print("⚠️  WARNING: May have data leakage!")

        return train, val, test
