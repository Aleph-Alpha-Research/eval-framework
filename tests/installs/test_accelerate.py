import accelerate


def test_accelerate_import() -> None:
    # Just instantiate core components to ensure they are importable
    version = accelerate.__version__
    assert version is not None


def main() -> None:
    test_accelerate_import()


if __name__ == "__main__":
    main()
