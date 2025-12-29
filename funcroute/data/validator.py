"""
Data validation utilities
"""

from typing import List, Dict, Set
from collections import Counter


class DataValidator:
    """Validate training data quality and detect potential issues"""

    def __init__(self):
        """Initialize validator"""
        pass

    def validate(
        self,
        data: List[Dict[str, str]],
        min_samples_per_tool: int = 10,
        warn_duplicates: bool = True,
        warn_imbalance: bool = True,
    ) -> Dict[str, any]:
        """
        Validate training data.

        Args:
            data: Training data samples
            min_samples_per_tool: Minimum samples required per tool
            warn_duplicates: Whether to warn about duplicate queries
            warn_imbalance: Whether to warn about class imbalance

        Returns:
            Validation report dict

        Example:
            >>> validator = DataValidator()
            >>> report = validator.validate(train_data)
            >>> if not report['is_valid']:
            ...     print(report['errors'])
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        if len(data) == 0:
            report["is_valid"] = False
            report["errors"].append("Data is empty")
            return report

        # Check tool distribution
        tool_counts = Counter(s["tool"] for s in data)
        report["stats"]["tool_counts"] = dict(tool_counts)
        report["stats"]["num_tools"] = len(tool_counts)
        report["stats"]["num_samples"] = len(data)

        # Check minimum samples per tool
        for tool, count in tool_counts.items():
            if count < min_samples_per_tool:
                report["warnings"].append(
                    f"Tool '{tool}' has only {count} samples "
                    f"(minimum recommended: {min_samples_per_tool})"
                )

        # Check for duplicates
        if warn_duplicates:
            queries = [s["query"] for s in data]
            duplicates = [q for q, count in Counter(queries).items() if count > 1]
            if duplicates:
                report["warnings"].append(
                    f"Found {len(duplicates)} duplicate queries"
                )
                report["stats"]["duplicate_queries"] = duplicates[:5]

        # Check for class imbalance
        if warn_imbalance and len(tool_counts) > 1:
            max_count = max(tool_counts.values())
            min_count = min(tool_counts.values())
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 3:
                report["warnings"].append(
                    f"Class imbalance detected: ratio {imbalance_ratio:.1f}:1 "
                    f"(max: {max_count}, min: {min_count})"
                )

        # Check query lengths
        query_lengths = [len(s["query"]) for s in data]
        report["stats"]["avg_query_length"] = sum(query_lengths) / len(query_lengths)
        report["stats"]["min_query_length"] = min(query_lengths)
        report["stats"]["max_query_length"] = max(query_lengths)

        # Check for very short queries
        short_queries = [s for s in data if len(s["query"]) < 3]
        if short_queries:
            report["warnings"].append(
                f"Found {len(short_queries)} very short queries (< 3 chars)"
            )

        return report

    def print_report(self, report: Dict[str, any]):
        """
        Print validation report.

        Args:
            report: Validation report from validate()
        """
        print("\n" + "=" * 80)
        print("DATA VALIDATION REPORT")
        print("=" * 80)

        # Stats
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {report['stats']['num_samples']}")
        print(f"  Unique tools: {report['stats']['num_tools']}")
        print(f"  Avg query length: {report['stats']['avg_query_length']:.1f} chars")
        print(f"  Min/Max query length: {report['stats']['min_query_length']}/{report['stats']['max_query_length']}")

        # Tool distribution
        print(f"\nTool Distribution:")
        for tool, count in sorted(report['stats']['tool_counts'].items()):
            percentage = (count / report['stats']['num_samples']) * 100
            print(f"  {tool:30s} {count:5d} samples ({percentage:5.1f}%)")

        # Errors
        if report["errors"]:
            print(f"\n❌ ERRORS ({len(report['errors'])}):")
            for error in report["errors"]:
                print(f"  - {error}")

        # Warnings
        if report["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        else:
            print(f"\n✅ No warnings")

        # Overall status
        print("\n" + "=" * 80)
        if report["is_valid"] and not report["warnings"]:
            print("✅ DATA IS VALID - Ready for training")
        elif report["is_valid"]:
            print("⚠️  DATA IS VALID - But has warnings")
        else:
            print("❌ DATA IS INVALID - Fix errors before training")
        print("=" * 80)

    def check_leakage(
        self,
        train_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
    ) -> bool:
        """
        Check for data leakage between train and test sets.

        Args:
            train_data: Training samples
            test_data: Test samples

        Returns:
            True if no leakage, False if leakage detected
        """
        train_queries = set(s["query"] for s in train_data)
        test_queries = set(s["query"] for s in test_data)

        overlap = train_queries & test_queries

        if overlap:
            print(f"\n❌ DATA LEAKAGE DETECTED!")
            print(f"   {len(overlap)} queries appear in both train and test")
            print(f"\nExamples:")
            for query in list(overlap)[:5]:
                print(f"  - {query}")
            return False
        else:
            print(f"\n✅ NO DATA LEAKAGE")
            print(f"   Train: {len(train_queries)} unique queries")
            print(f"   Test: {len(test_queries)} unique queries")
            print(f"   Overlap: 0")
            return True
