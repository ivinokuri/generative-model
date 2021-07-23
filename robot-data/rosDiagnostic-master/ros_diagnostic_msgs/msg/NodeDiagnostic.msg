string node
string host
string uri
string pid

# Statistics apply to this time window
time window_start
time window_stop

uint16 samples

uint16 threads

# Process CPU usage, as percentage of total local use
float32[] cpu_loads

# Virtual Memory use
float64[] virt_mems

# Real Memory Use
float64[] real_mems

