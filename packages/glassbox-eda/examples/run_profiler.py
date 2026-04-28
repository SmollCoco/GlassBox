#!/usr/bin/env python3
"""Small script to run DataProfiler on a CSV file."""

from __future__ import annotations

import argparse
import json

from GlassBox.eda.profiler import DataProfiler
from GlassBox.numpandas.io.csv import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a CSV file with GlassBox EDA")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument(
        "--html",
        default="eda_profile_report.html",
        help="Output HTML report path (default: eda_profile_report.html)",
    )
    args = parser.parse_args()

    df = read_csv(args.input_csv)

    profiler = DataProfiler(df)
    profiler.compute_profile()

    print(json.dumps(profiler.profile, indent=2, ensure_ascii=False, default=str))

    if args.html:
        profiler.generate_html_report(args.html)
        print(f"\nHTML report written to: {args.html}")


if __name__ == "__main__":
    main()
