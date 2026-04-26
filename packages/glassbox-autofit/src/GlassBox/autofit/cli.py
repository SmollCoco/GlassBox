import argparse
import json
import sys
import traceback

from GlassBox.autofit.core import autofit


def main():
    parser = argparse.ArgumentParser(description="GlassBox AutoFit CLI")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument(
        "--output",
        default="/results/best_model.pkl",
        help="Path where the fitted model pipeline will be saved",
    )
    args = parser.parse_args()

    try:
        # Run the end-to-end pipeline
        report, _ = autofit(
            csv_path=args.data,
            target_col=args.target,
            tuning=True,
            output_path=args.output,
        )
        # Print the final JSON report for the NemoClaw agent
        print(json.dumps(report, indent=4))

    except Exception as e:
        error_report = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_report, indent=4))
        sys.exit(1)


if __name__ == "__main__":
    main()
