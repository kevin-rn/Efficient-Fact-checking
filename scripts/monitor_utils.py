import time
from memory_profiler import memory_usage
import math
from pathlib import Path
import os

TIME_INTERVALS = (
    ('weeks', 604800),
    ('days', 86400),
    ('hours', 3600),
    ('minutes', 60),
    ('seconds', 1),
)

def get_dir_size(dir_path) -> int:
    """
    Calculate the total size of a directory and its subdirectories.

    Parameters:
        - dir_path (str): The path to the directory.

    Returns:
        The total size of the directory in bytes.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def format_size(total_size: int) -> str:
    """
    Format the size in bytes to a human-readable string.

    Parameters:
        - total_size (int): The size in bytes.

    Returns:
        The formatted size string.
    """
    if total_size == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(total_size, 1024)))
    p = math.pow(1024, i)
    s = round(total_size / p, 2)
    return f"{s} {size_name[i]}"

def format_time(seconds: int) -> str:
    """
    Format the time in seconds to a human-readable string.

    Parameters:
        - seconds (int): The time in seconds.

    Returns:
        The formatted time string.
    """
    if seconds < 0:
        raise ValueError("Time cannot be negative")

    result = []
    for name, count in TIME_INTERVALS:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append(f"{int(value)} {name}")
    return ', '.join(result)

def monitor(func):
    """
    Monitor the performance of a function, including elapsed time, data, output size, and memory usage.

    Parameters:
        - func (callable): The function to be monitored.

    Prints:
        Elapsed time, data size, output size, and maximum memory usage.
    """
    root = Path(__file__).parent.parent
    data_path = os.path.join(root, "baselines", "hover", "data", "hover")
    out_path = os.path.join(root, "baselines", "hover", "out", "hover", "exp1.0")

    data_before, out_before = get_dir_size(data_path), get_dir_size(out_path)
    mem_usage = [0.0]
    start_time = time.perf_counter()
    try:
        mem_usage = memory_usage(func)
    except:
        pass
    end_time = time.perf_counter()
    data_after, out_after = get_dir_size(data_path), get_dir_size(out_path)

    total_time = end_time - start_time
    total_data_folder = data_after - data_before
    total_output_folder = out_after - out_before

    print(f"elapsed time: {format_time(total_time)}")
    print(f"hover datasplit: {format_size(total_data_folder)}, model data: {format_size(total_output_folder)}")
    print(f"Maximum memory usage: {max(mem_usage):.3f} MiB")
