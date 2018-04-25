from __future__ import print_function

import sys
import subprocess
import os.path

def run(cmd):
    verbose = False
    if verbose:
        print('-'*80)
        print('Executing:', cmd)
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (stdoutdata, stderrdata) = process.communicate()
    lines = stdoutdata.strip()
    if verbose:
        if len(lines) > 0:
            for line in lines.split('\n'):
                print('|', line)
    return lines

def run_times(cmd):
    lines = run(cmd)
    times = []
    for line in lines.split('\n'):
        parts = line.split()
        if parts[0] == 'Time:':
            times.append(float(parts[1]))
        if line.find('Check matrix:') >= 0:
            if parts[2] != 'OK':
                print('Check matrix failed. Exiting.')
                exit(0)
    return times

n_runs = 100
n_threads = 4
n_threads_per_block = 128

print('%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s' % ('cases', 'algorithm', 't_CPU', 't_CPU*', 'gain', 't_CPU', 't_GPU', 'gain'))
print('-'*82)

for i in range(1, 8):
    n_cases = 10**i
    logfile = 'eventlog_' + str(n_cases) + '.pre'
    if not os.path.isfile(logfile):
        print('File not found:', logfile)
        exit(0)
    for alg in ['flow', 'handover', 'together']:
        print('%-10d %-10s' % (n_cases, alg), end='')
        cmd = './%s_cpu %d %d %s' % (alg, n_runs, n_threads, logfile)
        times = run_times(cmd)
        print('%-10.6f %-10.6f %-10.3f' % (times[0], times[1], 0 if times[1] == 0 else times[0]/times[1]), end='')
        cmd = './%s_gpu %d %d %s' % (alg, n_runs, n_threads_per_block, logfile)
        times = run_times(cmd)
        print('%-10.6f %-10.6f %-10.3f' % (times[0], times[1], 0 if times[1] == 0 else times[0]/times[1]))
    print('-'*82)
