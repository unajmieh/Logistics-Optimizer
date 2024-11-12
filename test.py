# --- CPU usage plots ------------------
# --- Memory usage plots ---------------
"""
import psutil  
import matplotlib.pyplot as plt  
from time import sleep  

def monitor_cpu_usage(duration_sec=60):  
    cpu_usage = []  # List to store CPU usage values  
    interval = 1  # Sampling interval in seconds  

    for _ in range(duration_sec):  
        cpu_percent = psutil.cpu_percent(interval=interval)  
        cpu_usage.append(cpu_percent)  
        sleep(interval)  

    return cpu_usage  

def monitor_memory_usage(duration_sec=60):  
    memory_usage = []  # List to store memory usage values  
    interval = 1  # Sampling interval in seconds  

    for _ in range(duration_sec):  
        mem = psutil.virtual_memory()  
        memory_usage.append(mem.percent)  # Append memory usage percentage  
        sleep(interval)  

    return memory_usage  

def monitor_disk_usage(duration_sec=60):  
    disk_usage = []  # List to store disk usage values  
    interval = 1  # Sampling interval in seconds  

    for _ in range(duration_sec):  
        disk = psutil.disk_usage('/')  
        disk_usage.append(disk.percent)  # Append disk usage percentage  
        sleep(interval)  

    return disk_usage  

# Duration for monitoring  
duration = 60  # Monitor for 60 seconds  

# Monitor CPU, Memory, and Disk usage  
cpu_usage_values = monitor_cpu_usage(duration)  
memory_usage_values = monitor_memory_usage(duration)  
disk_usage_values = monitor_disk_usage(duration)  

# Create plots  
plt.figure(figsize=(15, 5))  

# CPU Usage Plot  
plt.subplot(1, 3, 1)  
plt.plot(cpu_usage_values, color='blue')  
plt.xlabel("Time (seconds)")  
plt.ylabel("CPU Usage (%)")  
plt.title("CPU Usage Over Time")  
plt.grid(True)  

# Memory Usage Plot  
plt.subplot(1, 3, 2)  
plt.plot(memory_usage_values, color='green')  
plt.xlabel("Time (seconds)")  
plt.ylabel("Memory Usage (%)")  
plt.title("Memory Usage Over Time")  
plt.grid(True)  

# Disk Usage Plot  
plt.subplot(1, 3, 3)  
plt.plot(disk_usage_values, color='red')  
plt.xlabel("Time (seconds)")  
plt.ylabel("Disk Usage (%)")  
plt.title("Disk Usage Over Time")  
plt.grid(True)  

# Show all plots  
plt.tight_layout()  
plt.show()

# ---- Sensitivity Analysis -------------

num_iterations_values = [30, 50, 100, 150, 200]
Lambda_values = [1.5, 1.78, 2.0]
step_size_values = [1.1, 2.5, 2.8]
cost_values = []
results = []

for num_iterations in num_iterations_values:  
    for Lambda in Lambda_values:  
        for step_size in step_size_values:  
            best_nest, best_cost, convergence_costs = cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,  cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size) 
            
            # Clear cost_values before appending new values for each iteration  
            cost_values.clear()  
            for nest in best_nest:  # Assuming best_nest is a list of nests  
                cost_values.append(cost_calculator.calculate_cost_component2(nest))  
                
            # Collecting the results for each cost value  
            for cost_value in cost_values:  
                results.append({  
                    'num_iterations': num_iterations,   
                    'Lambda': Lambda,   
                    'step_size': step_size,   
                    'cost': cost_value  
                })

# Assuming `results` is your list of dictionaries created previously  
# Step 1: Convert the results into a DataFrame  
df_results = pd.DataFrame(results)  

# Step 2: Create the line plot  
plt.figure(figsize=(12, 6))  # Optional: Adjust the figure size  
sns.lineplot(data=df_results, x='num_iterations', y='cost', hue='Lambda', style='step_size', markers=True)  

# Step 3: Add titles and labels  
plt.title('Cost Analysis for Different Parameters')  
plt.xlabel('Number of Iterations')  
plt.ylabel('Cost')  
plt.legend(title='Lambda and Step Size')  
plt.grid(True)  

# Step 4: Show the plot  
plt.show() 
"""
# --- Robustness Analysis ---------------
#########################################


#--------------- thesis Table ---------------------------
"""
1) the best cost found by the algorithm 
2) the number of iterations of the algorithm 
3) the average cost of top found nests
4) the diversity of the found nests
5) the portion of the nests that have been successfully replaced 
6) measuring the uniqueness of the top solution 
7) the total convergence time of the algorithm 
8) the sample size confidence interval


"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,  cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size):
    NumberObjectionEvaluations = 0
    convergence_costs = []
    G = 0
    while NumberObjectionEvaluations < MaxNumberEvaluations and G < MaxNumberEvaluations:
        G += 1
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)
        for sublist in nests:
            nests.sort(key=lambda x: cost_calculator.calculate_cost_component2(sublist))

            for i in range(len(nests) // 2):
                X_i = nests[i]
                alpha = 1.5
                step_vector = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)
                X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                F_i = cost_calculator.calculate_cost_component2(X_i)
                if F_i > cost_calculator.calculate_cost_component2(X_k):
                    nests[i] = X_k
            for i in range(len(nests) // 2, len(nests)):
                X_i = nests[i]
                X_j = random.choice(nests[:len(nests) // 2])

                if X_i == X_j:
                    alpha = 1.5
                    step_vector_2 = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector_2)]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)
                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k
                else:
                    nest_x_i = np.array(X_i)
                    nest_x_j = np.array(X_j)
                    squared_diff = np.sum(np.abs(nest_x_i - nest_x_j))
                    euclidean_dist = np.sqrt(np.abs(squared_diff))
                    dx = int(euclidean_dist / GoldenRatio)
                    X_k = [int(coord + dx) for coord in X_i]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)


        X = np.array(nests)

        # Replace KMeans with Gaussian Mixture clustering  
        gmm = GaussianMixture(n_components=10, n_init=5, random_state=0).fit(X) 
        cluster_centers = gmm.means_  # Get the means of the Gaussian components  

        # Calculate distances from each nest to each cluster center  
        distances = [np.linalg.norm(np.array(nest) - cluster_center) for nest in nests for cluster_center in cluster_centers]  

        # Sort distances and extract indices  
        sorted_indices = np.argsort(distances)  

        # Get top 30 nests based on sorted distances  
        top_nests = [nests[i] for i in sorted_indices if i < len(nests)][:10]  

        # Calculate the best cost for the first nest in top_nests  
        best_cost = cost_calculator.calculate_cost_component2(top_nests[0])  
        convergence_costs.append(best_cost)  

        NumberObjectionEvaluations += 1
        
    similarities = cosine_similarity(nests, top_nests_3)
    return top_nests, best_cost, convergence_costs, similarities


top_nests_3, best_cost_3, convergence_costs_3, similarities  = cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,  cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size)

print("thesis table")
print("######")
# Assuming you've already executed the `cuckoo_search_with_cost` function
# and obtained the values for `top_nests_3`, `best_cost_3`, and `convergence_costs_3`.


# Assuming you have the initial nests (initial_nests_array) and top nests (top_nests_array)


# Calculate the average cosine similarity
uniqueness = 1 - similarities.mean()


# Calculate the requested metrics
num_iterations = MaxNumberEvaluations
avg_top_nests_cost = sum(convergence_costs_3) / len(convergence_costs_3)
diversity = np.std(convergence_costs_3)
num_successful_replacements = len(convergence_costs_3) - 1  # Subtract 1 for the initial best cost
uniqueness = 1 - similarities.mean()  # You need to define `similarities`
# You can add timing measurements for convergence time if needed

# Calculate sample size confidence interval (you need to define the sample size)
sample_size = len(convergence_costs_3)  # Define your actual sample size
confidence_interval = 1.96 * (diversity / np.sqrt(sample_size))

# Print the results
print("Best Cost:", best_cost_3)
print("Number of Iterations:", num_iterations)
print("Average Cost of Top Nests:", avg_top_nests_cost)
print("Diversity of Found Nests:", diversity)
print("Portion of Successfully Replaced Nests:", num_successful_replacements / num_iterations)
print("Uniqueness of Top Solution:", uniqueness)
print("Sample Size Confidence Interval:", confidence_interval)

#------------Thesis Table 2------- 
# 1) number of nests
# 2) number of iterations 
# 3) termination condition 
# 4) levy flight main parameter 
# 5) nest abandon rate 
# 6) objective function top cost
# 7) the running time 
# 8) convergence rate 
# 9) End timing 
print("thesis table 2")
import numpy as np  
import random  
import time  
from sklearn.mixture import GaussianMixture  
import time  
import random  
import numpy as np  
from sklearn.mixture import GaussianMixture  
from MainNestGeneration import num_internal_arrays, num_elements, max_value
import psutil  # Import the psutil library  

def cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,   
                             cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size):  
    NumberObjectionEvaluations = 0  
    convergence_costs = []  
    G = 0  
    
    # Start timing  
    start_time = time.time()  
    
    # Lists to store CPU and memory usage  
    cpu_usages = []  
    memory_usages = []  

    while NumberObjectionEvaluations < MaxNumberEvaluations and G < MaxNumberEvaluations:  
        G += 1  
        #nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value) 
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max) 
        
        for sublist in nests:  
            nests.sort(key=lambda x: cost_calculator.calculate_cost_component2(sublist))  

            # Perform local search on the first half of nests  
            for i in range(len(nests) // 2):  
                X_i = nests[i]  
                alpha = 1.5  
                step_vector = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)  
                X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]  
                F_i = cost_calculator.calculate_cost_component2(X_i)  
                if F_i > cost_calculator.calculate_cost_component2(X_k):  
                    nests[i] = X_k  

            # Handle crossover for the second half of nests  
            for i in range(len(nests) // 2, len(nests)):  
                X_i = nests[i]  
                X_j = random.choice(nests[:len(nests) // 2])  

                if X_i == X_j:  
                    alpha = 1.5  
                    step_vector_2 = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)  
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector_2)]  
                    F_k = cost_calculator.calculate_cost_component2(X_k)  
                    l = random.randint(0, len(nests) - 1)  
                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):  
                        nests[l] = X_k  
                else:  
                    nest_x_i = np.array(X_i)  
                    nest_x_j = np.array(X_j)  
                    squared_diff = np.sum(np.abs(nest_x_i - nest_x_j))  
                    euclidean_dist = np.sqrt(np.abs(squared_diff))  
                    dx = int(euclidean_dist / GoldenRatio)  
                    X_k = [int(coord + dx) for coord in X_i]  
                    F_k = cost_calculator.calculate_cost_component2(X_k)  
                    l = random.randint(0, len(nests) - 1)  

        # Replace KMeans with Gaussian Mixture clustering  
        gmm = GaussianMixture(n_components=10, n_init=5, random_state=0).fit(nests)   
        cluster_centers = gmm.means_  # Get the means of the Gaussian components  

        # Calculate distances from each nest to each cluster center  
        distances = [np.linalg.norm(np.array(nest) - cluster_center) for nest in nests for cluster_center in cluster_centers]  

        # Sort distances and extract indices  
        sorted_indices = np.argsort(distances)  

        # Get top nests based on sorted distances  
        top_nests = [nests[i] for i in sorted_indices if i < len(nests)][:10]  

        # Calculate the best cost for the first nest in top_nests  
        best_cost = cost_calculator.calculate_cost_component2(top_nests[0])  
        convergence_costs.append(best_cost)  

        # Record CPU and memory usage at the end of each iteration  
        cpu_usage = psutil.cpu_percent(interval=None)  
        memory_usage = psutil.virtual_memory().percent  
        cpu_usages.append(cpu_usage)  
        memory_usages.append(memory_usage)  

        NumberObjectionEvaluations += 1  

    # End timing  
    end_time = time.time()  
    running_time = end_time - start_time  

    # Additional metrics  
    convergence_rate = np.mean(np.diff(convergence_costs)) if len(convergence_costs) > 1 else 0  

    # Return results including CPU and memory usage  
    return top_nests, best_cost, convergence_costs, running_time, convergence_rate, cpu_usages, memory_usages
# Calling the function and printing results  
MaxNumberEvaluations = 66  # Example value, set this according to your needs  
num_nests = 20  # Example value, set this according to your needs  
num_vessels = 1  # Example value, set this according to your needs  
a_range = 10  # Example value, set this according to your needs  
b_range = 6   # Example value, set this according to your needs  
c_min = 2     # Example value, set this according to your needs  
c_max = 4     # Example value, set this according to your needs  
Lambda = 1.5   # Example value, set this according to your specific problem  
dimension = 5  # Example value, set this according to your specific problem  
step_size = 0.1  # Example value, set this according to your specific problem  

# Execute the cuckoo search  
top_nests, best_cost, convergence_costs, running_time, convergence_rate, cpu_usages, memory_usages  = cuckoo_search_with_cost(  
    MaxNumberEvaluations, num_nests, num_vessels, a_range, cost_calculator, custom_rng,   
    LocalSearch, Lambda, dimension, step_size  
)  

# Print the requested metrics  
print("1) Number of nests:", num_nests)  
print("2) Number of iterations (evaluations):", num_iterations)  
print("3) Termination condition: Maximum evaluations reached")  
print("4) Levy flight main parameter (Lambda):", Lambda)  # Assuming Lambda is defined in your context  
print("5) Nest abandon rate: N/A for Cuckoo Search without explicit abandonment rate")  
print("6) Objective function top cost:", best_cost_3)  
print("7) The running time (in seconds):", running_time)  
print("8) Convergence rate (Average Improvement per Evaluation):", convergence_rate)  
print("9) cpu usages:", cpu_usages)
print("9) memory usages:", memory_usages)
