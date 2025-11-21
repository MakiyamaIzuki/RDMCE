#!/usr/bin/env python3
import subprocess, time, sys

import re
def get_mem():
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=memory.used,memory.total',
        '--format=csv,noheader,nounits'
    ], text=True).splitlines()[0]
    used, total = map(int, re.findall(r'\d+', out))
    return used, total

peak = 0
while True:
    used, total = get_mem()
    if used > peak:
        peak = used
        print(f'Peak CUDA memory: {peak} MiB / {total} MiB', flush=True)
    time.sleep(0.1)