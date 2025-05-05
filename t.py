import sys
import time

# Example for progress bar
for i in range(101):
    print(f"\rProgress: {i}%", end='', flush=True)
    time.sleep(0.1)