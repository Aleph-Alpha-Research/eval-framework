import argparse
import logging
import re
from datetime import datetime

from packaging.version import Version

CHANGELOG_PATH = "CHANGELOG.md"
UNRELEASED_MARKER = "## [Unreleased]"

logger = logging.getLogger(__name__)


def read_changelog(changelog_path: str) -> str:
    with open(changelog_path) as file:
        return file.read()


def write_changelog(changelog_path: str, content: str) -> None:
    with open(changelog_path, "w") as file:
        file.write(content)


def update_changelog(changelog_path: str, today: str) -> str:
    content = read_changelog(changelog_path)
    current_version = find_current_version(content)
    next_version = determine_next_version(content, current_version)

    # Regex to find the [Unreleased] section and capture everything until the next section
    pattern = r"## \[Unreleased\]\s*(.*?)(?=## \[|$)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return "No [Unreleased] section found."

    # Extract the part before and after the [Unreleased] section
    before_unreleased = content[: match.start()]
    after_unreleased = content[match.end() :]
    unreleased_content = match.group(0)
    # Remove markers from unreleased content
    cleaned_content = re.sub(r"\s*\[(MAJOR|MINOR|PATCH)\]", "", unreleased_content)
    if cleaned_content.replace(UNRELEASED_MARKER, "").strip() == "":
        return "No Content in [Unreleased] section."
    # Prepare new version header and reset the [Unreleased] section
    new_version_header = f"## [{next_version}] - {today}\n"
    cleaned_content = cleaned_content.replace(f"{UNRELEASED_MARKER}\n", new_version_header)
    new_unreleased_section = f"{UNRELEASED_MARKER}\n\n"

    # Construct the updated content
    updated_content = before_unreleased + new_unreleased_section + cleaned_content + after_unreleased
    write_changelog(changelog_path, updated_content)
    return updated_content  # Return the updated content


def get_content_of_version(changelog_path: str, version: str) -> str:
    content = read_changelog(changelog_path)
    pattern = rf"## \[{version}\]\s*(.*?)(?=## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"No [version] section found for {version}")
    return match.group(0)


def find_current_version(changelog_content: str) -> str:
    # Regex to find version numbers in the format ## [x.y.z] - DATE
    version_pattern = re.compile(r"## \[(\d+\.\d+\.\d+)\]")
    versions = version_pattern.findall(changelog_content)
    if not versions:
        return "0.0.0"  # Default to "0.0.0" if no versions are found

    sorted_versions = sorted(versions, key=lambda x: Version(x), reverse=True)
    return sorted_versions[0]


def determine_next_version(content: str, current_version: str = "0.0.0") -> str:
    v = Version(current_version)
    if "[MAJOR]" in content:
        next_version = f"{v.major + 1}.0.0"
    elif "[MINOR]" in content:
        next_version = f"{v.major}.{v.minor + 1}.0"
    else:
        next_version = f"{v.major}.{v.minor}.{v.micro + 1}"

    return str(Version(next_version))


def check_version(given_version: str) -> str:
    v1 = Version(given_version)
    v2 = Version(find_current_version(read_changelog(CHANGELOG_PATH)))
    return str(v1 if v1 < v2 else v2)


def main() -> None:
    parser = argparse.ArgumentParser(description="A simple CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    check_parser = subparsers.add_parser(
        "check", help="Compares the Changelog version with a given version and returns the smaller one"
    )
    check_parser.add_argument("version_string", type=str, help="The version to check")

    subparsers.add_parser("release", help="Bump the changelog version")
    subparsers.add_parser("current", help="Get the changelog version")
    content_parser = subparsers.add_parser("content", help="Get the content of a given version")
    content_parser.add_argument("version_string", type=str, help="The version to check")
    args = parser.parse_args()

    if args.command == "check":
        logger.info(check_version(args.version_string))

    elif args.command == "release":
        logger.info("Releasing the tool ...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        result = update_changelog(CHANGELOG_PATH, current_date)
        logger.info(result)
    elif args.command == "current":
        logger.info(check_version(find_current_version(read_changelog(CHANGELOG_PATH))))
    elif args.command == "content":
        logger.info(get_content_of_version(CHANGELOG_PATH, args.version_string))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
