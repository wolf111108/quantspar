"""Collect Qwen Bitlet BCE columns and conditional latency with paper defaults."""
from .profile_ebb import main as profile_main


def main(argv=None):
    return profile_main(argv, backend="bitlet")


if __name__ == "__main__":
    main()
