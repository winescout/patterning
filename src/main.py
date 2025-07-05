import argparse
from cli.commands import add_spike_commands


def main():
    parser = argparse.ArgumentParser(description="Video Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    add_spike_commands(subparsers)  # Add spike commands
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
