"""Export artifacts (ckpt, onnx, torchscript) placeholder"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    print(f"Would export artifacts for {args.ckpt}")

if __name__ == '__main__':
    main()
