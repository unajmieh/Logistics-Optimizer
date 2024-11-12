# this is for the search space exploration ---- 

import numpy as np  
import matplotlib.pyplot as plt  
import psutil  
import time  

# Simulated operation function (replace with your actual function)  
def simulate_operation():  
    # Simulate some CPU work  
    time.sleep(0.1)  # Simulate computation time  
    # Simulate some memory allocation  
    _ = [x * 2 for x in range(10000)]  # Allocate some memory (10,000 integers)  

# Function to monitor resource usage during iterations  
def monitor_resources(num_iterations):  
    memory_usages = []  
    disk_usages = []  

    for iteration in range(num_iterations):  
        # Simulate some operation  
        simulate_operation()  

        # Get the memory usage  
        memory_info = psutil.virtual_memory()  
        memory_usages.append(memory_info.percent)  

        # Get the disk usage (for the root partition)  
        disk_info = psutil.disk_usage('/')  
        disk_usages.append(disk_info.percent)  

    return memory_usages, disk_usages  

# Parameters  
num_iterations = 50  # Set your number of iterations  

# Monitor resources  
memory_usages, disk_usages = monitor_resources(num_iterations)  

# Prepare x values for plotting  
x_values = np.arange(num_iterations)  

# Plotting memory usage  
plt.figure(figsize=(12, 6))  

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot  
plt.plot(x_values, memory_usages, label='Memory Usage (%)', color='green')  
plt.title('Memory Usage during Iterations')  
plt.xlabel('Iteration')  
plt.ylabel('Memory Usage (%)')  
plt.axhline(y=np.mean(memory_usages), color='r', linestyle='--', label='Average Memory Usage')  
plt.legend()  
plt.grid()  

# Plotting disk usage  
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot  
plt.plot(x_values, disk_usages, label='Disk Usage (%)', color='orange')  
plt.title('Disk Usage during Iterations')  
plt.xlabel('Iteration')  
plt.ylabel('Disk Usage (%)')  
plt.axhline(y=np.mean(disk_usages), color='r', linestyle='--', label='Average Disk Usage')  
plt.legend()  
plt.grid()  

# Adjust layout and show plot  
plt.tight_layout()  
plt.show()
