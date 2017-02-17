# h2d_overlapping
benchmarking the overlapping for multiple cuda streams

## Author
Leiming Yu

* Email: ylm@ece.neu.edu
* Twitter: @yu_leiming
* Blog: http://www1.coe.neu.edu/~ylm/

## Background
When running multiple cuda streams to explore the concurrency for data transfer, the actual starting point of launching next cuda stream depends on the cuda driver and gpu device.

This boilerplate helps identify the starting point.

<img src="h2d_ovlp.png" height="500"></img>

## How to get it work?
* Create your benchmark. Here, I use vector add kernel.
* Create the executable.
* Specify the options through shell script and dump the trace file using **nvprof**.
* Run the python script transfer_ovlp.py.
* Finally, it will print out the data size for each vector and the starting piont timing.

```bash
N = 17400, d2h-h2d overlap : 0.028192 (ms)
N = 2409300, h2d-h2d overlap : 3.158431 (ms)
 ```
 
p.s., the data size is increased with 100 floats per step. You can decrease it for more accurate timing.
