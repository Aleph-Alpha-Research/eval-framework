import argparse
import json

from colorama import Back, Fore, Style, init

# Initialize colorama
init()


def colorize_json(data: str | dict | list, indent: int = 0, highlights: list = [], strip_keys: list = []) -> str:
    """
    Colorize JSON keys and values for terminal output with indentation.
    """
    indentation = " " * indent
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if key in strip_keys:
                continue

            key_str = f'{Fore.BLUE}"{key}"{Style.RESET_ALL}'
            if key in highlights:
                key_str = f"{Back.YELLOW}{key_str}{Style.RESET_ALL}"

            items.append(f"\n{indentation}{key_str}: {colorize_json(value, indent + 4, highlights, strip_keys)}")
        return "{" + ",".join(items) + "\n" + indentation + "}"
    elif isinstance(data, list):
        items = [colorize_json(item, indent + 4, highlights, strip_keys) for item in data]
        return "[\n" + ("\n" + indentation).join(items) + "\n" + indentation + "]"
    else:
        data_str = (
            f'{Fore.GREEN}"{data}"{Style.RESET_ALL}'
            if isinstance(data, str)
            else f"{Fore.CYAN}{json.dumps(data)}{Style.RESET_ALL}"
            if isinstance(data, bool)
            else f"{Fore.MAGENTA}null{Style.RESET_ALL}"
            if data is None
            else f"{Fore.YELLOW}{data}{Style.RESET_ALL}"
        )
        if str(data) in highlights:
            data_str = f"{Back.YELLOW}{data_str}{Style.RESET_ALL}"
        return data_str


def jsonl_to_colored_json(filepath: str, highlights: list, strip_keys: list) -> None:
    """
    Converts JSONL file to pretty printed and colorized JSON.
    """
    try:
        with open(filepath, "r") as file:
            for line in file:
                data = json.loads(line)
                colored_output = colorize_json(data, 4, highlights, strip_keys)
                print(colored_output)
                print()
    except FileNotFoundError:
        print(f"{Fore.RED}File not found: {filepath}{Style.RESET_ALL}")
    except json.JSONDecodeError:
        print(f"{Fore.RED}Error decoding JSON from file: {filepath}{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to pretty-printed and colorized JSON.")
    parser.add_argument("filepath", metavar="FILE", type=str, help="Path to the JSONL file.")
    parser.add_argument(
        "--highlight",
        type=lambda x: x.split(","),
        default=[],
        help="Comma-separated list of keys or values to highlight.",
    )
    parser.add_argument(
        "--strip", type=lambda x: x.split(","), default=[], help="Comma-separated list of keys to exclude from output."
    )

    args = parser.parse_args()

    jsonl_to_colored_json(args.filepath, args.highlight, args.strip)
