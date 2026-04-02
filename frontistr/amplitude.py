# txtファイルに振幅データを書き出す
# python3 amplitude.py 

import math

def write_amplitude_to_txt(filename, amplitude_data):
    with open(filename, 'w') as f:
        f.write("\n!AMPLITUDE, NAME=AMP0\n")
        for time, amplitude in amplitude_data:
            f.write(f"{amplitude:.10e}, {time:.10e}\n")
        f.write("!END\n")

if __name__ == "__main__":
    n = 2001
    start, stop = 0, 20
    time_values = [start + i * (stop - start) / (n - 1) for i in range(n)]
    amplitude_values = [math.sin(t) for t in time_values]

    amplitude_data = list(zip(time_values, amplitude_values))
    write_amplitude_to_txt("amplitude_data.txt", amplitude_data)