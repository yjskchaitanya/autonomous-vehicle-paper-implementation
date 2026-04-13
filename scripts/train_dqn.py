from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_reimpl.config import Config
from paper_reimpl.training import set_seed, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.load(args.config).raw
    set_seed(int(config["seed"]))
    train_dqn(config)


if __name__ == "__main__":
    main()
