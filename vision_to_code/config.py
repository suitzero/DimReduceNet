from dataclasses import dataclass
from pathlib import Path

@dataclass
class FactoryConfig:
    resolution: int = 64
    output_dir: Path = Path("dataset")
    seed: int = 42

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
