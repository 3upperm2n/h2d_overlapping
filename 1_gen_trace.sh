#!/bin/bash
streams=2
devid=0
nvprof --print-gpu-trace --csv ./contention $streams $devid "$@" 2> trace.csv
