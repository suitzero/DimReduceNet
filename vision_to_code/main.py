import argparse
import random
from pathlib import Path
from vision_to_code.config import FactoryConfig
from vision_to_code.factory import Factory

def main():
    parser = argparse.ArgumentParser(description="Vision-to-Code Data Factory")
    parser.add_argument("--resolution", type=int, default=64, help="Image resolution (e.g. 64)")
    parser.add_argument("--count", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    config = FactoryConfig(
        resolution=args.resolution,
        output_dir=Path(args.output),
        seed=args.seed
    )

    factory = Factory(config)

    print(f"Starting generation: {args.count} samples at {args.resolution}x{args.resolution}...")
    factory.generate_batch(count=args.count)
    print(f"Done! Data saved to {args.output}/{args.resolution}x{args.resolution}/")

if __name__ == "__main__":
    main()
