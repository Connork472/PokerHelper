#!/usr/bin/env python3
"""
PokerHelper Launcher
Choose between GUI or CLI interface
"""

import sys
import os

def main():
    print("🎯 PokerHelper Launcher")
    print("=" * 30)
    print("Choose your interface:")
    print("1. GUI Interface (Recommended)")
    print("2. Command Line Interface")
    print("3. Exit")
    print("=" * 30)
    
    while True:
        try:
            choice = input("Select option (1-3): ").strip()
            
            if choice == "1":
                print("Starting GUI interface...")
                try:
                    import main
                    main.main()
                except ImportError as e:
                    print(f"GUI not available: {e}")
                    print("Falling back to CLI interface...")
                    import cli_main
                    cli_main.main()
                break
                
            elif choice == "2":
                print("Starting CLI interface...")
                import cli_main
                cli_main.main()
                break
                
            elif choice == "3":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-3.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
