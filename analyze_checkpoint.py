#!/usr/bin/env python3
"""
Automated checkpoint analysis tool.

Analyzes both clipped and unclipped model checkpoints for negative parameter values.
Outputs:
  - TSV-formatted summary for easy spreadsheet copy-paste
  - Detailed per-layer statistics for verification
  - Summary statistics
"""
import argparse
import os
import sys

# Add project root to path to allow running as script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.checkpoint_utils import (
    get_model_config,
    find_checkpoint_files,
    load_checkpoint_and_analyze
)


def format_number(num):
    """Format number with thousands separator."""
    return f"{num:,}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model checkpoints for negative parameter values"
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        help="Full path to checkpoint folder"
    )

    args = parser.parse_args()

    # Get folder path (from arg or interactive prompt)
    if args.folder_path:
        folder_path = args.folder_path
    else:
        folder_path = input("Enter checkpoint folder path: ").strip()
        if not folder_path:
            print("Error: Folder path cannot be empty")
            sys.exit(1)

    try:
        # Get model config
        print("Loading model configuration...")
        model_config = get_model_config()

        # Find checkpoint files
        print(f"Searching for checkpoints in: {folder_path}")
        unclipped_path, clipped_path = find_checkpoint_files(folder_path)

        unclipped_name = os.path.basename(unclipped_path)
        clipped_name = os.path.basename(clipped_path)

        print(f"Found checkpoints:")
        print(f"  - {unclipped_name}")
        print(f"  - {clipped_name}")
        print()

        # Analyze both checkpoints
        results = {}

        for ckpt_path, ckpt_name in [(unclipped_path, unclipped_name),
                                      (clipped_path, clipped_name)]:
            print(f"Loading {ckpt_name}...")
            model, aggregated, full_results = load_checkpoint_and_analyze(
                ckpt_path, model_config, verbose=False
            )
            results[ckpt_name] = {
                'model': model,
                'aggregated': aggregated,
                'full_results': full_results
            }
            print(f"  Loaded {format_number(aggregated['total_params'])} parameters")
            print(f"  Found {format_number(aggregated['negative_params'])} negative parameters")
            print()

        # Output Section 1: Copy-Paste Summary (TSV)
        print("=" * 80)
        print("COPY-PASTE SECTION (for spreadsheet)")
        print("=" * 80)
        print("Checkpoint Type\tNegative Ratio\tMax Negative Value\tTotal Params\tNegative Params")

        for ckpt_name in [unclipped_name, clipped_name]:
            agg = results[ckpt_name]['aggregated']
            print(
                f"{ckpt_name}\t"
                f"{agg['ratio']:.4f}\t"
                f"{agg['max_negative']:.4f}\t"
                f"{agg['total_params']}\t"
                f"{agg['negative_params']}"
            )

        print()
        print("(Select the data rows above and paste into your spreadsheet)")
        print()

        # Output Section 2: Detailed Per-Layer Statistics
        for ckpt_name in [unclipped_name, clipped_name]:
            print("=" * 80)
            print(f"DETAILED ANALYSIS: {ckpt_name}")
            print("=" * 80)

            # Re-run check_non_negativity with verbose=True to print details
            from utils.non_negativity import check_non_negativity
            check_non_negativity(results[ckpt_name]['model'], verbose=True)
            print()

        # Output Section 3: Summary Statistics
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Folder: {folder_path}")
        print()

        for label, ckpt_name in [("Unclipped", unclipped_name),
                                  ("Clipped", clipped_name)]:
            agg = results[ckpt_name]['aggregated']
            print(f"{label}:")
            print(f"  - Total parameters: {format_number(agg['total_params'])}")
            print(f"  - Negative parameters: {format_number(agg['negative_params'])}")
            print(f"  - Ratio: {agg['ratio']:.4f} ({agg['ratio']*100:.2f}%)")
            print(f"  - Max negative value: {agg['max_negative']:.4f}")
            print()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
