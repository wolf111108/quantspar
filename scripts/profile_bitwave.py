"""Collect BitWave alone using the same Qwen runner as joint profiling."""
from .profile_ebb import main as _main


def main(argv=None):
    return _main(argv, backend="bitwave")


if __name__ == "__main__":
    main()
