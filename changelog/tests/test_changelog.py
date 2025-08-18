from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from changelog.changelog import (
    determine_next_version,
    get_content_of_version,
    read_changelog,
    update_changelog,
    write_changelog,
)


@pytest.fixture()
def header() -> str:
    return (
        "# Changelog\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\n"
        "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
        "<!-- Here's an example template:\n"
        "## [Unreleased]\n"
        "### Added\n"
        "- For new features.\n"
        "### Changed\n"
        "- For changes in existing functionality.\n"
        "### Deprecated\n"
        "- For soon-to-be removed features.\n"
        "### Removed\n"
        "- For now removed features.\n"
        "### Fixed\n"
        "- For any bug fixes.\n"
        "### Security\n"
        "- In case of vulnerabilities.\n"
        "-->\n"
    )


@pytest.fixture
def temp_changelog(tmp_path: Path) -> str:
    # Create a temporary changelog file
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_content = "## [Unreleased]\n\n- Initial commit\n"
    changelog_path.write_text(changelog_content)
    return str(changelog_path)


@pytest.fixture
def read_changelog_filled() -> str:
    changelog_content = """
    ## [2.0.0]
    - Major update

    ## [1.1.0]
    - Added new feature

    ## [1.0.0]
    - Initial release

    """
    return changelog_content


def test_read_changelog(temp_changelog: str) -> None:
    content = read_changelog(temp_changelog)
    assert content == "## [Unreleased]\n\n- Initial commit\n"


def test_write_changelog(temp_changelog: str) -> None:
    new_content = "## [Unreleased]\n\n- Added new feature\n"
    write_changelog(temp_changelog, new_content)

    updated_content = read_changelog(temp_changelog)
    assert updated_content == new_content


@pytest.mark.parametrize(
    "data, current_version, expected_version",
    [
        pytest.param(
            "## [Unreleased]\n- [MAJOR] Complete overhaul\n- [MINOR] New sorting feature\n- Bug fix",
            "2.4.1",
            "3.0.0",
            id="major_with_minor_and_implicit_patch",
        ),
        pytest.param(
            "## [Unreleased]\n- [MINOR] Added search feature\n- [PATCH] Fixed typo",
            "0.9.8",
            "0.10.0",
            id="explicit_minor_and_patch",
        ),
        pytest.param(
            "## [Unreleased]\n- [MINOR] New analytics panel\n- Fixed crash issue",
            "1.15.2",
            "1.16.0",
            id="minor_with_implicit_patch",
        ),
        pytest.param(
            "## [Unreleased]\n- [MAJOR] API v2 rollout\n- [MAJOR] Database schema changes",
            "1.0.0",
            "2.0.0",
            id="multiple_explicit_majors",
        ),
        pytest.param(
            "## [Unreleased]\n- [MAJOR] Migration to cloud\n- Updated logging",
            "3.2.5",
            "4.0.0",
            id="major_with_implicit_patch",
        ),
        pytest.param(
            "## [Unreleased]\n- [MINOR] Migration to cloud\n- Updated logging",
            "0.0.0",
            "0.1.0",
            id="minor_from_no_previous_version",
        ),
    ],
)
def test_version_determination_simplified(data: str, current_version: str, expected_version: str, header: str) -> None:
    with patch("builtins.open", mock_open(read_data=f"{header}{data}")):
        assert determine_next_version(data, current_version) == expected_version


def test_update_changelog_automatic_version_detection() -> None:
    header = (
        "# Changelog\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\n"
        "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
    )
    original_content = (
        f"{header}"
        "## [Unreleased]\n"
        "### Added\n"
        "- [MINOR] New feature\n"
        "## [1.2.3] - 2023-06-01\n"
        "- Changes\n"
        "## [1.2.2] - 2023-05-01\n"
        "- Changes\n"
    )
    expected_content = (
        f"{header}"
        "## [Unreleased]\n\n"
        f"## [1.3.0] - 2024-07-15\n"
        "### Added\n"
        "- New feature\n"
        "## [1.2.3] - 2023-06-01\n"
        "- Changes\n"
        "## [1.2.2] - 2023-05-01\n"
        "- Changes\n"
    )

    with (
        patch("changelog.changelog.read_changelog", return_value=original_content),
        patch("changelog.changelog.write_changelog") as mock_write,
    ):
        result = update_changelog("CHANGELOG.md", "2024-07-15")
        mock_write.assert_called_once_with("CHANGELOG.md", expected_content)

        assert result == expected_content, "The returned updated content does not match the expected content."


def test_first_version_creation() -> None:
    header = (
        "# Changelog\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\n"
        "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
    )
    original_content = f"{header}## [Unreleased]\n- [MINOR] Initial release of the project\n"
    expected_content = f"{header}## [Unreleased]\n\n## [0.1.0] - 2024-07-15\n- Initial release of the project\n"

    with (
        patch("changelog.changelog.read_changelog", return_value=original_content),
        patch("changelog.changelog.write_changelog") as mock_write,
    ):
        result = update_changelog("CHANGELOG.md", "2024-07-15")
        mock_write.assert_called_once_with("CHANGELOG.md", expected_content)
        assert result == expected_content, "The returned updated content does not match the expected content."


def test_get_content_of_version_existing_version(read_changelog_filled: str) -> None:
    with patch("changelog.changelog.read_changelog", return_value=read_changelog_filled):
        content = get_content_of_version("dummy_path", "1.1.0")
    assert content.strip() == "## [1.1.0]\n    - Added new feature"


def test_get_content_of_version_non_existing_version(read_changelog_filled: str) -> None:
    with pytest.raises(ValueError, match="No \[version\] section found for 3.0.0"):  # type: ignore
        with patch("changelog.changelog.read_changelog", return_value=read_changelog_filled):
            get_content_of_version("dummy_path", "3.0.0")


def test_get_content_of_version_empty_changelog(read_changelog_filled: str) -> None:
    with pytest.raises(ValueError, match="No \[version\] section found for 1.0.0"):
        with patch("changelog.changelog.read_changelog", return_value=""):
            get_content_of_version("dummy_path", "1.0.0")
