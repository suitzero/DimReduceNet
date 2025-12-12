import json
import matplotlib.pyplot as plt
from pathlib import Path
from .generators.gen_2d import Gen2D
from .generators.gen_math import GenMath
from .generators.gen_fractal import GenFractal
from .generators.gen_3d import Gen3D

class Factory:
    def __init__(self, config):
        self.config = config
        self.generators = {
            "2d": Gen2D(config),
            "math": GenMath(config),
            "fractal": GenFractal(config),
            "3d": Gen3D(config)
        }

    def generate_batch(self, count=10):
        """Generates a batch of data across all categories."""
        # Ensure base output dir exists
        base_dir = self.config.output_dir / f"{self.config.resolution}x{self.config.resolution}"
        base_dir.mkdir(parents=True, exist_ok=True)

        generated_count = 0
        categories = list(self.generators.keys())

        while generated_count < count:
            cat = categories[generated_count % len(categories)]
            gen = self.generators[cat]

            try:
                code, img, meta = gen.generate()
                self._save_sample(base_dir, code, img, meta)
                generated_count += 1
                if generated_count % 10 == 0:
                    print(f"Generated {generated_count}/{count} samples...")
            except Exception as e:
                print(f"Error generating sample for {cat}: {e}")
                continue

    def _save_sample(self, base_dir, code, img, meta):
        # Structure: dataset/64x64/category/uuid/
        sample_id = meta["id"]
        category = meta["category"]

        sample_dir = base_dir / category / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save Code
        with open(sample_dir / "code.py", "w") as f:
            f.write(code)

        # Save Image
        plt.imsave(sample_dir / "image.png", img, cmap='gray')

        # Save Metadata
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
