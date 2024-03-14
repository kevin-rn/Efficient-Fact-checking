import math
import os
import time
from pathlib import Path
from threading import Thread

import nvidia_smi
import psutil

# Week, days, hours, minutes and seconds.
TIME_INTERVALS = (
    ("w", 604800),
    ("d", 86400),
    ("h", 3600),
    ("m", 60),
    ("s", 1),
)
SIZE_INTERVALS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")


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
    if total_size <= 0:
        return "0.0 B"
    i = int(math.floor(math.log(total_size, 1024)))
    p = math.pow(1024, i)
    s = round(total_size / p, 2)
    return f"{s} {SIZE_INTERVALS[i]}"


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
            result.append(f"{int(value)}{name}")
    return ":".join(result) if result else "> 1s"


class ProcessMonitor(Thread):
    """
    Monitor the performance of a function, including elapsed time, data, output size, and memory usage.
    """

    def __init__(self, dataset=None, *args, **kwargs):
        """
        Initialises Thread class.
        """
        super().__init__(*args, **kwargs)

        # Setup arrays to keep track of values
        self.cpu_usage = []
        self.mem_usage = []
        self.GPU = [{"util": [], "mem": []}, {"util": [], "mem": []}]
        self.measure_dataset = dataset
        # Path to directories
        root = Path(__file__).parent.parent.parent
        if self.measure_dataset:
            self.data_path = os.path.join(root, "data", dataset)
            self.out_path = os.path.join(root, "out", dataset)
            # Measure before running function
            self.data_before = get_dir_size(dir_path=self.data_path)
            self.out_before = get_dir_size(dir_path=self.out_path)
        self.start_time = time.perf_counter()
        # Setup GPU monitoring before running
        nvidia_smi.nvmlInit()
        self.deviceCount = nvidia_smi.nvmlDeviceGetCount()

    def __enter__(self):
        return self

    def run(self):
        """
        Starts thread run.
        """
        currentProcess = psutil.Process()
        self.running = True
        while self.running:
            # Measure CPU and RAM usage
            cpu = currentProcess.cpu_percent(interval=1) / psutil.cpu_count()
            self.cpu_usage.append(cpu)
            self.mem_usage.append(currentProcess.memory_info().rss)

            # Measure GPU usage
            for i in range(self.deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                self.GPU[i]["util"].append(util.gpu)
                self.GPU[i]["mem"].append(mem.used)

    def print_gpu_stats(self):
        """
        Helper function to print out GPU monitor stats.
        """
        for device_i in range(self.deviceCount):
            util = self.GPU[device_i]["util"]
            mem = self.GPU[device_i]["mem"]
            util_stats = max(util, default=0) - min(util, default=0)
            mem_stats = max(mem, default=0) - min(mem, default=0)

            print(
                f"GPU {device_i} - Util {util_stats:.1f}% - Mem: {format_size(mem_stats)}"
            )

    def __exit__(self, type, value, traceback):
        """
        Stops monitoring thread and prints out stats.
        """
        self.running = False
        end_time = time.perf_counter()
        if self.measure_dataset:
            data_after = get_dir_size(self.data_path)
            out_after = get_dir_size(self.out_path)

        # Substract the before measurement to avoid overlapping processes being measured.
        total_time = end_time - self.start_time
        cpu = max(self.cpu_usage, default=0) - min(self.cpu_usage, default=0)
        mem = max(self.mem_usage, default=0) - min(self.mem_usage, default=0)

        # Sleep few seconds for files to finish processing
        if self.measure_dataset:
            time.sleep(2)
            total_data_folder = data_after - self.data_before
            total_output_folder = out_after - self.out_before

        print(f"Elapsed time: {format_time(total_time)}")
        if self.measure_dataset:
            print(
                f"datasplit: {format_size(total_data_folder)}, model data: {format_size(total_output_folder)}"
            )
        print(f"CPU : {cpu:.1f}%")
        print(f"RAM : {format_size(mem)}")
        self.print_gpu_stats()
