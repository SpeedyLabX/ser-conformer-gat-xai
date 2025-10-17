"""Lightweight train CLI placeholder"""
import argparse
from serxai.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    set_seed()
    print(f"Would load config {args.config}")
    if args.dry_run:
        print("Dry run: exiting after setup")

if __name__ == '__main__':
    main()
