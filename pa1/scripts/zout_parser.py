#! /usr/bin/python
#####################################################################
# EE282 Programming Assignment 1:
# Optimization of Matrix Multiplication
#
# Created by: mgao12	04/14/2014
#####################################################################

import sys, os
from collections import namedtuple
from zsim_parse import *

PerCoreStats = namedtuple('PerCoreStat', ['instrs', 'cycles', 'l1d_misses', 'l2_misses'])
Stats = namedtuple('Stats', ['dim', 'MFLOPS', 'cores', 'l3_misses'])

################################################################################

def parse_zsim_out(dim, zoutfile, num_cores, l3_num_banks):
    # Get the parser
    zout_parser = ZsimOutParser(zoutfile)
    # Strings
    core_str_prefix = 'nehalem.nehalem-'
    l1d_str_prefix = 'l1d.l1d-'
    l2_str_prefix = 'l2.l2-'
    l3_str_prefix = 'l3.l3-0b'
    # Get repeats. Must be the same as driver.c
    if dim < 128:
        repeats = 100
    elif dim < 512:
        repeats = 10
    else:
        repeats = 1
    # Parse
    cores = []
    max_cycles = 0
    for cid in range(num_cores):
        core_str = core_str_prefix + str(cid)
        instrs = zout_parser.get_counters_and(core_str, 'instrs', one) / repeats
        cycles = zout_parser.get_counters_and(core_str, 'cycles', one) / repeats
        l1d_str = l1d_str_prefix + str(cid)
        l2_str = l2_str_prefix + str(cid)
        l1d_misses = zout_parser.get_cache_miss(l1d_str) / repeats
        l2_misses = zout_parser.get_cache_miss(l2_str) / repeats
        cores.append(PerCoreStats(instrs, cycles, l1d_misses, l2_misses))
        max_cycles = max(max_cycles, cycles)
    l3_misses = 0
    for bid in range(l3_num_banks):
        l3_b_str = l3_str_prefix + str(bid)
        l3_misses += zout_parser.get_cache_miss(l3_b_str) / repeats
    # Estimate MFLOPS assuming naive matmul
    mflops = 2.0 * (dim**3) / (max_cycles / 2930.0)
    return Stats(dim, mflops, cores, l3_misses)

################################################################################

if len(sys.argv) < 2:
    print 'Usage: ' + sys.argv[0] + ' [sim dir]'
    sys.exit(1)

sim_dir = os.path.abspath(sys.argv[1])

# Get and parse results
all_stats = []
for d in os.listdir(sim_dir):
    abs_path = os.path.join(sim_dir, d)
    # Skip the file/dir that is not a zsim dir
    if not os.path.isdir(abs_path):
        continue
    out_file = os.path.join(abs_path, 'zsim.out')
    if not os.path.exists(out_file):
        continue
    with open(out_file, 'r') as fout:
        content = fout.readlines()
    # Skip if simulation doesn't finish (# lines in zsim.out < 10)
    if len(content) < 10:
        continue
    # Matrix size
    sz = d.split('-')[-1]
    if not sz.isdigit():
        print 'Can\'t parse the matrix size from the output directory ' \
            + d + '! Did you change it?'
        sys.exit(1)
    sz = int(sz)
    stats = parse_zsim_out(sz, out_file, 4, 4)
    all_stats.append(stats)

# Sort and output
all_stats = sorted(all_stats, key=lambda Stats: Stats[0])
print '\nSimulation results for ' + os.path.basename(sim_dir)
print 'Each measurement is average per iteration.'
print 'MFLOPS numbers are estimated assuming a naive matmul.\n'
header = ' '.join(['Size'.rjust(5), 'MFLOPS'.rjust(10), ' '*3, \
        'Cycles'.rjust(12), 'Instrs'.rjust(12), 'L1D Misses'.rjust(12), \
        'L2 Misses'.rjust(12), 'L3 Misses'.rjust(12)])
print header
for stats in all_stats:
    msg = '{0:5d} {1:10.3f}    {2} {3} {4} {5} {6:12d}\n'.format( \
            stats.dim, stats.MFLOPS, ' '*12, ' '*12, ' '*12, ' '*12, stats.l3_misses)
    for cid in range(4):
        core_stats = stats.cores[cid]
        msg += ' '*20 + '{0:12d} {1:12d} {2:12d} {3:12d}\n'.format( \
                core_stats.cycles, core_stats.instrs, core_stats.l1d_misses, core_stats.l2_misses)
    print msg

