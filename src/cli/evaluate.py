"""Evaluate CLI placeholder"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=False)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--export-attn", action="store_true")
    args = parser.parse_args()
    print("Evaluate placeholder")

if __name__ == '__main__':
    main()
