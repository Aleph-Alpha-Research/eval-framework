from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

import eval_framework.metrics.completion.code_execution.sandbox as sandbox_module
from eval_framework.metrics.completion.code_execution.sandbox import get_or_create_pool


class TestGetOrCreatePool:
    """Tests for get_or_create_pool caching behaviour."""

    @pytest.fixture(autouse=True)
    def reset_pools(self) -> Iterator[None]:
        """Isolate each test by clearing the module-level pool cache."""
        original = sandbox_module._pools.copy()
        sandbox_module._pools.clear()
        yield
        sandbox_module._pools.clear()
        sandbox_module._pools.update(original)

    def test_same_args_returns_cached_pool(self) -> None:
        # GIVEN create_pool_manager is mocked to return a fresh MagicMock each call
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN get_or_create_pool is called twice with identical arguments
            pool_first = get_or_create_pool(image="python:3.13-slim", packages=["numpy"])
            pool_second = get_or_create_pool(image="python:3.13-slim", packages=["numpy"])

            # THEN create_pool_manager is only invoked once and both calls yield the same object
            mock_create.assert_called_once()
            assert pool_first is pool_second

    def test_different_images_create_separate_pools(self) -> None:
        # GIVEN create_pool_manager is mocked
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN get_or_create_pool is called with two different images
            pool_a = get_or_create_pool(image="python:3.12-slim")
            pool_b = get_or_create_pool(image="python:3.13-slim")

            # THEN create_pool_manager is called once per unique image and returns distinct objects
            assert mock_create.call_count == 2
            assert pool_a is not pool_b

    def test_different_packages_create_separate_pools(self) -> None:
        # GIVEN create_pool_manager is mocked
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN get_or_create_pool is called with the same image but different package lists
            pool_a = get_or_create_pool(image="python:3.13-slim", packages=["numpy"])
            pool_b = get_or_create_pool(image="python:3.13-slim", packages=["pandas"])

            # THEN two distinct pools are created
            assert mock_create.call_count == 2
            assert pool_a is not pool_b

    def test_no_packages_and_empty_packages_are_same_key(self) -> None:
        # GIVEN create_pool_manager is mocked
        # packages=None and packages=[] are both falsy, so they share the same cache key
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            pool_none = get_or_create_pool(image="python:3.13-slim", packages=None)
            pool_empty = get_or_create_pool(image="python:3.13-slim", packages=[])

            assert mock_create.call_count == 1
            assert pool_none is pool_empty

    def test_same_dockerfile_returns_cached_pool(self) -> None:
        # GIVEN create_pool_manager is mocked
        dockerfile = "FROM python:3.13-slim\nRUN pip install numpy"
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN get_or_create_pool is called twice with the same dockerfile
            pool_first = get_or_create_pool(dockerfile=dockerfile)
            pool_second = get_or_create_pool(dockerfile=dockerfile)

            # THEN the pool is only created once and the same object is returned
            mock_create.assert_called_once()
            assert pool_first is pool_second

    def test_different_dockerfiles_create_separate_pools(self) -> None:
        # GIVEN create_pool_manager is mocked
        dockerfile_a = "FROM python:3.12-slim"
        dockerfile_b = "FROM python:3.13-slim"
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN get_or_create_pool is called with two distinct dockerfiles
            pool_a = get_or_create_pool(dockerfile=dockerfile_a)
            pool_b = get_or_create_pool(dockerfile=dockerfile_b)

            # THEN a separate pool is created for each dockerfile
            assert mock_create.call_count == 2
            assert pool_a is not pool_b

    def test_dockerfile_with_packages_cached_separately(self) -> None:
        # GIVEN create_pool_manager is mocked
        dockerfile = "FROM python:3.13-slim"
        target = "eval_framework.metrics.completion.code_execution.sandbox.create_pool_manager"
        with patch(target, side_effect=lambda **_: MagicMock()) as mock_create:
            # WHEN called with the same dockerfile but different package lists
            pool_no_pkgs = get_or_create_pool(dockerfile=dockerfile)
            pool_with_pkgs = get_or_create_pool(dockerfile=dockerfile, packages=["numpy"])

            # THEN two distinct pools are created, one per (dockerfile, packages) combination
            assert mock_create.call_count == 2
            assert pool_no_pkgs is not pool_with_pkgs

            # AND a third call with the same dockerfile+packages reuses the cached pool
            pool_with_pkgs_again = get_or_create_pool(dockerfile=dockerfile, packages=["numpy"])
            assert mock_create.call_count == 2
            assert pool_with_pkgs_again is pool_with_pkgs
