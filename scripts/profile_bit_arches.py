"""One calibration/inference pass collecting Bitlet and BitWave independently."""
from .profile_ebb import main as _main


def main(argv=None):
    return _main(argv, backend="bitlet_bitwave")


if __name__ == "__main__":
    main()
