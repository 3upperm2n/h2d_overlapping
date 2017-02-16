# h2d_overlapping
benchmarking the overlapping for multiple cuda streams

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

p.s., the data size is increased with 100 floats per step. You can decrease it for more accurate timing.
