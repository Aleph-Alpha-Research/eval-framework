from eval_framework.tasks.utils import BIG_CODE_BENCH_PACKAGE_MAPPING, CallableSerializer


class TestBigCodeBenchDataset:
    def test_mapping_coverage(self) -> None:
        """Test the coverage of our mapping against a sample of common packages."""
        common_packages = [
            # Standard library
            "os",
            "sys",
            "json",
            "datetime",
            "re",
            "math",
            # External packages
            "numpy",
            "pandas",
            "matplotlib",
            "sklearn",
            "tensorflow",
            "PIL",
            "requests",
            "bs4",
            "flask",
            "django",
        ]

        for pkg in common_packages:
            assert pkg in BIG_CODE_BENCH_PACKAGE_MAPPING, f"Package {pkg} not in mapping"


def test_fn_recover() -> None:
    def fn(x: int) -> int:
        return x * 2

    serializer = CallableSerializer()
    encoded_fn = serializer.encode(fn)
    decoded_fn = serializer.decode(encoded_fn)
    assert decoded_fn(2) == fn(2)
