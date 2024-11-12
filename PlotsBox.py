import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

# Create sample data for demonstration  
data = {  
    'Function': ['Griewank', 'Schwefel', 'Rosenbrock', 'Rastrigin', 'Main Objective'],  
    'Cuckoo Search': [11955.107, 1.48065e103, 1.825013e22, 2.09357e97, 0.8751],  
    'Enhanced Cuckoo Search': [210.51755, 1.614385e103, 6.589505e21, 7.977405e95, 0.786692],  
    'Enhanced Cuckoo Search with Gaussian mixture clustering': [1.5945291, 4.0711622, 14.501383, 2.005455, 2.479310],   
    'Particle Swarm Optimization': [2.35822e-20, 6.19206e102, 7.39788e124, 7.020037e50, 0.100863]  
}  

# Create a DataFrame  
df = pd.DataFrame(data)  
df.set_index('Function', inplace=True)  

# Initialize a figure with subplots: 1 row for performance bar plots, 1 row for box plots  
fig, axes = plt.subplots(2, len(df.columns), figsize=(25, 10))  # 2 rows, number of algorithms as columns  

# Bar plots for each algorithm  
for i, algorithm in enumerate(df.columns):  
    # Bar plot  
    bars = axes[0, i].bar(df.index, df[algorithm], color='navy', alpha=0.7)  
    axes[0, i].set_title(f'Performance of {algorithm}', fontsize=10)  
    axes[0, i].set_ylabel('Objective Function Value', fontsize=10)  
    axes[0, i].set_xlabel('Functions', fontsize=10)  
    axes[0, i].set_yscale('log')  # Set Y scale to logarithmic  
    axes[0, i].grid(axis='y')  

    # Add value labels on top of each bar  
    for bar in bars:  
        yval = bar.get_height()  
        axes[0, i].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2e}', va='bottom', ha='center', fontsize=8)  

    # Box and whisker plots for the same data  
    axes[1, i].boxplot(df[algorithm], patch_artist=True, boxprops=dict(facecolor='purple', color='purple'),  
                         medianprops=dict(color='navy'), whiskerprops=dict(color='purple'),   
                         capprops=dict(color='purple'), flierprops=dict(marker_color='purple'))  

    axes[1, i].set_title(f'Cost Distribution of {algorithm}', fontsize=10)  
    axes[1, i].set_ylabel('Cost Values', fontsize=10)  
    axes[1, i].set_xlabel('Algorithms', fontsize=10)  

# Adjust layout  
plt.tight_layout()  
# Show the plot  
plt.show()