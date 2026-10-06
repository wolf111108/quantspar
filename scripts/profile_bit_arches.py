"""One calibration/inference pass collecting Bitlet, BitWave and Slim-Llama."""
from .profile_ebb import main as _main


def main(argv=None):
    return _main(argv, backend="bit_arches")


if __name__ == "__main__":
    main()
