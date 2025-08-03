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
    print("Version: 2.0.0 (Deliverable 2)")
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


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation - Deliverable 2")
    print()
    print("Usage:")
    print("  python3 main.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print("  --test-scheduling-with-metrics  Run scheduling tests with performance metrics")
    print()
    print("Once started, type 'help' for available shell commands")
    print()
    print("Quick Start - Process Scheduling:")
    print("  1. Configure algorithm:    scheduler config rr 2")
    print("  2. Add processes:          scheduler addprocess task1 5")
    print("  3. Start scheduler:        scheduler start")
    print("  4. Monitor status:         scheduler status")
    print("  5. View metrics:           scheduler metrics")


def run_scheduling_tests():
    """Run the scheduling tests with performance metrics"""
    print("Running scheduling tests with performance metrics...")
    
    # Import test functions
    from test_scheduling_with_metrics import (
        test_round_robin_configurable_time_slice,
        test_priority_with_time_simulation,
        test_preemption_with_time_simulation,
        test_early_completion_behavior
    )
    
    # Redirect all output to a text file
    output_file = "test_scheduling_with_metrics_output.txt"
    
    # Save original stdout
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to file
        with open(output_file, 'w') as f:
            sys.stdout = f
            
            print("Advanced Shell - Scheduling Test Suite with Performance Metrics")
            print("==============================================================")
            print()
            print("Testing Constraints:")
            print("1. Time slice is configurable and user-specified")
            print("2. Processes complete early if possible (removed from queue)")
            print("3. Process execution simulated with time.sleep()")
            print("4. Performance metrics tracked for all tests")
            print()
            
            try:
                test_round_robin_configurable_time_slice()
                test_priority_with_time_simulation()
                test_preemption_with_time_simulation()
                test_early_completion_behavior()
                
                print("All tests completed successfully!")
                print("The scheduling algorithms are working correctly with configurable time slices.")
                print("All test files and directories have been created in the 'test/' directory.")
                print()
                print("Key Features Demonstrated:")
                print("- Configurable time slices (0.5s, 1.0s, 2.0s)")
                print("- Early completion when processes finish before time slice")
                print("- Time simulation using actual command execution")
                print("- Priority-based preemption with time simulation")
                print("- Comprehensive performance metrics collection")
                
            except KeyboardInterrupt:
                print("\nTests interrupted by user")
            except Exception as e:
                print(f"Test error: {e}")
                
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Test output has been saved to: {output_file}")
        
        # Generate performance report
        from performance_metrics import performance_tracker
        performance_tracker.generate_report("performance_metrics_report.txt")
        print("Performance metrics report has been saved to: performance_metrics_report.txt")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 2",
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

    parser.add_argument(
        '--test-scheduling-with-metrics',
        action='store_true',
        help='Run scheduling tests with performance metrics'
    )

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.help:
        show_help()
        return

    if args.test_scheduling_with_metrics:
        run_scheduling_tests()
        return

    # Create and start the shell
    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            print("Scheduler modules available:", hasattr(shell.command_handler, 'process_scheduler'))
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