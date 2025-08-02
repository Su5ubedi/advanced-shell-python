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
    print("Version: 3.0.0 (Deliverable 3)")
    print("Build: Development")
    print()
    print("Features:")
    print("- Basic shell functionality (Deliverable 1)")
    print("- Built-in commands")
    print("- Process management")
    print("- Job control")
    print("- Process scheduling algorithms (Deliverable 2)")
    print("  * Round-Robin Scheduling")
    print("  * Priority-Based Scheduling")
    print("- Performance metrics")
    print("- Real-time process monitoring")
    print("- Command piping and redirection (Deliverable 3)")
    print("- User authentication system")
    print("- File permissions and access control")


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation - Deliverable 3")
    print()
    print("Usage:")
    print("  python3 main.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print()
    print("Once started, you will be prompted to log in.")
    print("Available users:")
    print("  admin/admin123    - Administrator (full access)")
    print("  user/user123      - Standard user (limited access)")
    print("  guest/guest123    - Guest user (very limited access)")
    print()
    print("Type 'help' for available shell commands")
    print()
    print("Quick Start - Authentication:")
    print("  login admin admin123    # Login as administrator")
    print("  whoami                  # Show current user")
    print("  logout                  # Logout current user")
    print()
    print("Quick Start - Piping:")
    print("  ls | grep txt           # List files containing 'txt'")
    print("  cat file.txt | sort     # Display sorted file contents")
    print("  ls | grep .py | wc -l   # Count Python files")
    print()
    print("Quick Start - Process Scheduling:")
    print("  1. Configure algorithm:    scheduler config rr 2")
    print("  2. Add processes:          scheduler addprocess task1 5")
    print("  3. Start scheduler:        scheduler start")
    print("  4. Monitor status:         scheduler status")
    print("  5. View metrics:           scheduler metrics")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 3",
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
            print("Scheduler modules available:", hasattr(shell.command_handler, 'process_scheduler'))
            print("Auth system available:", hasattr(shell.command_handler, 'auth_system'))
            print("Pipe handler available:", hasattr(shell.command_handler, 'pipe_handler'))
            # Additional debug setup could go here

        # Run the shell
        shell.run()

    except KeyboardInterrupt:
        print("\nShell interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()