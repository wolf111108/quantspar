"""Collect Slim-Llama alone with the shared FP8/W4 Qwen runner."""
from .profile_ebb import main as _main


def main(argv=None):
    return _main(argv, backend="slimllama")


if __name__ == "__main__":
    main()
