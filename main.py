#!/usr/bin/env python3
"""
main.py - Entry point for Advanced Shell Simulation
"""

import argparse
import sys
from shell import Shell


def show_version():
    """Show version information"""
    print("Advanced Shell Simulation")
    print("Version: 1.0.0 (Deliverable 1)")
    print("Build: Development")
    print()
    print("Features:")
    print("- Basic shell functionality")
    print("- Built-in commands")
    print("- Process management")
    print("- Job control")


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation")
    print()
    print("Usage:")
    print("  python main.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print()
    print("Once started, type 'help' for available shell commands")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 1",
        add_help=False  # We'll handle help ourselves
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )

    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help information'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.help:
        show_help()
        return

    # Create and start the shell
    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            # Additional debug setup could go here

        # Run the shell
        shell.run()

    except KeyboardInterrupt:
        print("\nShell interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()