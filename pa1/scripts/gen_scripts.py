#! /usr/bin/python
#####################################################################
# EE282 Programming Assignment 1:
# Optimization of Matrix Multiplication
#
# Created by: mgao12	04/14/2014
#####################################################################

import sys, os
import datetime  # for timestamp
import stat      # for chmod


if len(sys.argv) < 3:
    print 'Usage: ' + sys.argv[0] + ' [top dir] [cfg template]'
    sys.exit(1)

top_dir = os.path.abspath(sys.argv[1])
#top_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
template = sys.argv[2]

app = 'matmul'
test_sizes = [4, 16, 64, 256, 1024]

def make_cmd(basic_cmd, param):
    return basic_cmd + ' -s ' + param

# timestamp: HHMMSS-Month-Day-Year
time = datetime.datetime.now().strftime('%H%M%S-%b-%d-%Y')

app_dir = os.path.join(top_dir, 'matmul')
sim_dir = os.path.join(top_dir, 'sim' + '-' + time)

basic_cmd = os.path.join(app_dir, app)
cfg_prefix = os.path.basename(template).split('.')[0] + '-' + app

with open(template, 'r') as tfin:
    config = tfin.read()

for size in test_sizes:
    n = str(size)
    cfg_name = cfg_prefix + '-' + n
    cfg_dir = os.path.join(sim_dir, cfg_name)
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir);
    cfg_file = os.path.join(cfg_dir, cfg_name + '.cfg')
    cmd = make_cmd(basic_cmd, n)
    with open(cfg_file, 'w') as cfgout:
        cfgout.write(config)
        cfgout.write('\n\nprocess0 = {\n')
        cfgout.write('    command = \"' + cmd + '\";\n')
        cfgout.write('    startFastForwarded = true;\n')
        cfgout.write('    syncedFastForward = false;\n')
        cfgout.write('};\n')

run_script = os.path.join(sim_dir, 'run_script.sh')
with open(run_script, 'w') as rscout:
    rscout.write('#! /bin/bash\n\n')
    rscout.write('for n in ' + ' '.join([ str(sz) for sz in test_sizes ]) + '\ndo\n')
    rscout.write('    cd ' + cfg_prefix + '-$n\n')
    rscout.write('    zsim ' + cfg_prefix + '-$n.cfg\n')
    rscout.write('    cd ..\n')
    rscout.write('done\n')
st = os.stat(run_script)
os.chmod(run_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)



