import cProfile
import pstats
import time
import os
import runpy
from datetime import datetime
# ===== Configuration =====
TARGET_SCRIPT = r"main_script_parallel_grasp.py"          
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("results", "runtime_record")
OUTPUT_STATS_FILE = f"runtime_results_{timestamp}.txt"
SHOW_TOP = 20                              # Show only the top N most time-consuming custom functions
# =========================

script_abs_path = os.path.abspath(TARGET_SCRIPT)

def is_own_function(key):
    """
    key: (filename, lineno, funcname)
    Keep only functions/methods defined in the target script file (including class methods, closures, etc.).
    cProfile records the actual file name when run via runpy.run_path.
    """
    filename, _, _ = key
    try:
        return os.path.abspath(filename) == script_abs_path
    except Exception:
        return False
    

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, OUTPUT_STATS_FILE)

start_time = time.time()
pr = cProfile.Profile()

try:
    pr.enable()
    # Run the target script by file path to retain the correct file name in profiling data
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")
    pr.disable()
except SystemExit:
    # Handle sys.exit() in the target script gracefully
    pr.disable()
finally:
    total_time = time.time() - start_time

# Generate profiling statistics and keep only custom functions
stats = pstats.Stats(pr)
own_functions = {k: v for k, v in stats.stats.items() if is_own_function(k)}

# Sort and output results
sorted_funcs = sorted(
    own_functions.items(),
    key=lambda kv: kv[1][3],  # kv[1] = (cc, nc, tt, ct, callers); [3] is cumulative time (cumtime)
    reverse=True
)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Total runtime: {total_time:.6f} seconds\n\n")
    f.write(f"Custom functions by cumulative time (Top {SHOW_TOP}):\n")
    for idx, (func_key, stat) in enumerate(sorted_funcs[:SHOW_TOP], start=1):
        filename, lineno, funcname = func_key
        cc, nc, tt, ct, callers = stat
        f.write(
            f"{idx:02d}. {funcname}  Cumulative time: {ct:.6f} s  "
            f"Calls: {nc}  Location: {os.path.basename(filename)}:{lineno}\n"
        )

print(f"\nTotal runtime: {total_time:.6f} seconds")
print(f"Custom function profiling saved to: {output_path}")
print(f"Total functions recorded: {len(own_functions)} (filtered by file)")
