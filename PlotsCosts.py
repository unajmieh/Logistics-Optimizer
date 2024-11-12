import numpy as np  
import matplotlib.pyplot as plt  

# Original data  
data = {  
    "Cuckoo Search": [11955.107, 1.48065e103, 1.825013e22, 2.09357e97, 0.8751],  
    "Enhanced Cuckoo Search": [210.51755, 1.614385e103, 6.589505e21, 7.977405e95, 0.786692],  
    "GMM Cuckoo Search": [1.5945291, 4.0711622, 14.501383, 2.005455, 2.479310],  # no transformation  
    "Particle Swarm Optimization": [2.35822e-20, 6.19206e102, 7.39788e124, 7.020037e50, 0.100863],  
    "Estimation of Distribution": [0.4155923, 1.416252e103, 3.159257e226, 1.302159e102, 0.1436279],  
}  

# Logarithmic transformation for only some algorithms  
c = 1e-10  # small constant to avoid log(0)  
log_data = []  
for algorithm_name, algorithm_values in data.items():  
    if algorithm_name != "GMM Cuckoo Search":  
        log_data.append([np.log(x + c) for x in algorithm_values])  
    else:  
        log_data.append(algorithm_values)  # No transformation for GMM  

# Create the box and whisker plot with custom colors  
plt.figure(figsize=(10, 6))  
box = plt.boxplot(log_data, labels=data.keys(), patch_artist=True)  

# Define colors from navy blue to purple  
colors = ['#4f09db', '#4f09db', '#4f09db', '#4f09db', '#4f09db']  # Example gradient colors  
for patch, color in zip(box['boxes'], colors):  
    patch.set_facecolor(color)  
    patch.set_edgecolor('black')  
    patch.set_linewidth(1)  

# Additional aesthetics  
plt.title("Box and Whisker Plot of Algorithm Average and Median of the Costs")  
plt.ylabel("Transformed Log Cost")  
plt.grid(axis='y')  

# Show individual values for better comparison  
for i, algorithm_values in enumerate(log_data):  
    plt.scatter([i+1]*len(algorithm_values), algorithm_values, color='black', alpha=0.5)  

plt.show()