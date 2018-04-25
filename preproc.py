from __future__ import print_function

import re
import sys
import struct

def fix(value):
    return re.sub('[^0-9a-zA-Z]', '_', value)

if len(sys.argv) < 6:
    print('Usage: python %s 0 1 2 3 eventlog.csv' % sys.argv[0])
    print('0: column index for caseid')
    print('1: column index for task')
    print('2: column index for user')
    print('3: column index for timestamp')
    print('Column concatenations (e.g. 5+6) are also supported')
    print('Column names (e.g. A, B, ..., Z) are also supported')
    exit(0)

if '0' <= sys.argv[1][0] <= '9':
    col_caseid    = [int(col) for col in sys.argv[1].split('+')]
    col_task      = [int(col) for col in sys.argv[2].split('+')]
    col_user      = [int(col) for col in sys.argv[3].split('+')]
    col_timestamp = [int(col) for col in sys.argv[4].split('+')]
elif 'A' <= sys.argv[1][0] <= 'Z':
    col_caseid    = [ord(col)-ord('A') for col in sys.argv[1].split('+')]
    col_task      = [ord(col)-ord('A') for col in sys.argv[2].split('+')]
    col_user      = [ord(col)-ord('A') for col in sys.argv[3].split('+')]
    col_timestamp = [ord(col)-ord('A') for col in sys.argv[4].split('+')]
else:
    print('Column format not recognized. Exiting.')
    exit(0)

input_log_file = sys.argv[-1]
print('Reading:', input_log_file)
fin = open(input_log_file, 'r')

headers = None
log = []

for line in fin:
    line = line.strip()
    if len(line) == 0:
        continue
    parts = line.split(';')
    if headers == None:
        caseid = '+'.join([parts[col] for col in col_caseid])
        task = '+'.join([parts[col] for col in col_task])
        user = '+'.join([parts[col] for col in col_user])
        timestamp = '+'.join([parts[col] for col in col_timestamp])
        headers = [caseid, task, timestamp]
        print('Using column "%s" as caseid' % caseid)
        print('Using column "%s" as task' % task)
        print('Using column "%s" as user' % user)
        print('Using column "%s" as timestamp' % timestamp)
        print('Reading data...')
    else:
        caseid = ''.join([parts[col] for col in col_caseid])
        task = ''.join([parts[col] for col in col_task])
        user = ''.join([parts[col] for col in col_user])
        timestamp = ''.join([parts[col] for col in col_timestamp])
        log.append((caseid, task, user, timestamp))

fin.close()

print('Sorting event log...')

log.sort(key=lambda event: (event[0], event[-1]))

print('Finding case positions...')

pos_caseid = dict()
for i, (caseid, task, user, timestamp) in enumerate(log):
    if caseid not in pos_caseid:
        pos_caseid[caseid] = [i, i]
    pos_caseid[caseid][1] += 1

print('Collecting values...')

caseid_values = set()
caseid_len = 0
task_values = set()
task_len = 0
user_values = set()
user_len = 0
for (caseid, task, user, timestamp) in log:
    caseid_values.add(caseid)
    if len(caseid) > caseid_len:
        caseid_len = len(caseid)
    task_values.add(task)
    if len(task) > task_len:
        task_len = len(task)
    user_values.add(user)
    if len(user) > user_len:
        user_len = len(user)

print('Case ids:', len(caseid_values))
print('Tasks:', len(task_values))
print('Users:', len(user_values))

print('Sorting values...')

caseid_values = sorted(list(caseid_values))
task_values = sorted(list(task_values))
user_values = sorted(list(user_values))

print('Creating maps...')

caseid_map = {caseid:i for i, caseid in enumerate(caseid_values)}
task_map = {task:i for i, task in enumerate(task_values)}
user_map = {user:i for i, user in enumerate(user_values)}

if input_log_file.endswith('.csv'):
    output_data_file = input_log_file.replace('.csv', '.pre')
else:
    output_data_file = input_log_file + '.pre'
print('Writing:', output_data_file)
fout = open(output_data_file, 'wb')

fout.write(struct.pack('i', len(log)))
for (caseid, task, user, timestamp) in log:
    fout.write(struct.pack('i', caseid_map[caseid]))
    fout.write(struct.pack('i', task_map[task]))
    fout.write(struct.pack('i', user_map[user]))

fout.write(struct.pack('i', len(caseid_values)))
fout.write(struct.pack('i', caseid_len+1))
for caseid in caseid_values:
    fout.write(struct.pack(str(caseid_len+1)+'s', fix(caseid)))

fout.write(struct.pack('i', len(task_values)))
fout.write(struct.pack('i', task_len+1))
for task in task_values:
    fout.write(struct.pack(str(task_len+1)+'s', fix(task)))

fout.write(struct.pack('i', len(user_values)))
fout.write(struct.pack('i', user_len+1))
for user in user_values:
    fout.write(struct.pack(str(user_len+1)+'s', fix(user)))

for caseid in sorted(pos_caseid.keys()):
    fout.write(struct.pack('i', pos_caseid[caseid][0]))
    fout.write(struct.pack('i', pos_caseid[caseid][1]))

fout.close()
