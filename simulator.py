from __future__ import print_function

import sys
import simpy
import random
import datetime

env = simpy.Environment()

random.seed()

def random_duration(mu, sigma):
    r = random.normalvariate(mu, sigma)
    if r < 0.0:
        r = 0.0
    return r

t0 = datetime.datetime.now()

def current_time():
    t = t0 + datetime.timedelta(seconds=env.now)
    return t

tasks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
users = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']

params = dict()
params[tasks[0]] = (24.0, 12.0)
params[tasks[1]] = (48.0, 24.0)
params[tasks[2]] = (24.0, 12.0)
params[tasks[3]] = (24.0, 12.0)
params[tasks[4]] = (96.0, 48.0)
params[tasks[5]] = (24.0, 12.0)
params[tasks[6]] = (96.0, 48.0)
params[tasks[7]] = (48.0, 24.0)

probs = dict()
probs[tasks[0]] = [0.4, 0.6, 0. , 0. , 0. , 0. , 0. , 0. ]
probs[tasks[1]] = [0. , 0. , 0.7, 0.3, 0. , 0. , 0. , 0. ]
probs[tasks[2]] = [0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. ]
probs[tasks[3]] = [0. , 0. , 0. , 0. , 0.8, 0.2, 0. , 0. ]
probs[tasks[4]] = [0. , 0. , 0. , 0. , 0. , 0. , 0.6, 0.4]
probs[tasks[5]] = [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5]
probs[tasks[6]] = [0. , 0. , 0. , 0. , 0.3, 0.7, 0. , 0. ]
probs[tasks[7]] = [0.6, 0.4, 0. , 0. , 0. , 0. , 0. , 0. ]

def draw_user(task):
    r = random.random()
    p = 0.0
    for i in range(0, len(probs[task])):
        p += probs[task][i]
        if r < p:
            return users[i]

resources = dict()
for user in users:
    resources[user] = simpy.Resource(env, capacity=1)

def new_event(caseid, task, user):
    print(';'.join([str(caseid), task, user, current_time().strftime('%Y-%m-%d %H:%M:%S')]))

def user_proc(caseid, task, user):
    req = resources[user].request()
    yield req
    (mu, sigma) = params[task]
    duration = random_duration(mu, sigma)
    yield env.timeout(duration)
    new_event(caseid, task, user)
    resources[user].release(req)

def task_proc(caseid, task):
    user = draw_user(task)
    yield env.process(user_proc(caseid, task, user))

def branch1_proc(caseid):
    yield env.process(task_proc(caseid, tasks[4]))
    yield env.process(task_proc(caseid, tasks[5]))

def branch2_proc(caseid):
    yield env.process(task_proc(caseid, tasks[6]))

def instance_proc(caseid, tstart):
    yield env.timeout(tstart)
    yield env.process(task_proc(caseid, tasks[0]))
    yield env.process(task_proc(caseid, tasks[1]))
    r = random.random()
    if r < 0.25:
        yield env.process(task_proc(caseid, tasks[2]))
        return
    yield env.process(task_proc(caseid, tasks[3]))
    yield env.process(branch1_proc(caseid)) & env.process(branch2_proc(caseid))
    yield env.process(task_proc(caseid, tasks[7]))

if len(sys.argv) != 2:
    print('Usage: %s number_of_cases' % sys.argv[0])
    exit(0)

N = int(sys.argv[1])

tstart = 0.0
for caseid in range(1, N+1):
    r = random_duration(120.,60.)
    tstart += r
    env.process(instance_proc(caseid, tstart))

print('caseid;task;user;timestamp')

env.run()
