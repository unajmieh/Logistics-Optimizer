import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
from sklearn.mixture import GaussianMixture  
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


# Original Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]

# New Data
new_costs = [1.2964818586720162, 1.667964983231757, 1.8405206274384565, 2.4454385483949457, 
             2.6289952648181285, 2.7037709795579286, 3.0528928800442934, 3.414389272794696, 
             4.019114145951091, 4.061721465751667]

new_costs_1 = []

new_costs_2 = []

# X-axis positions (Nest indices)
x_original = np.arange(1, len(costs) + 1)
x_new = np.arange(len(costs) + 1, len(costs) + len(new_costs) + 1)

# Function to find Pareto front
def pareto_frontier(Xs, Ys, maxX=True, maxY=False):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_front_X = [pair[0] for pair in p_front]
    p_front_Y = [pair[1] for pair in p_front]
    return p_front_X, p_front_Y

# Find Pareto front for original and new data
p_front_X_original, p_front_Y_original = pareto_frontier(x_original, costs)
p_front_X_new, p_front_Y_new = pareto_frontier(x_new, new_costs)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_original, costs, color='blue', label='Original Costs')
plt.scatter(x_new, new_costs, color='green', label='New Costs')
plt.plot(p_front_X_original, p_front_Y_original, color='red', linestyle='--', label='Pareto Front (Original)')
plt.plot(p_front_X_new, p_front_Y_new, color='orange', linestyle='--', label='Pareto Front (New)')

# Add value labels
for i, cost in enumerate(costs):
    plt.annotate(f'{cost:.2f}', (x_original[i], costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')
for i, cost in enumerate(new_costs):
    plt.annotate(f'{cost:.2f}', (x_new[i], new_costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with Pareto Fronts')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


#--------Initial Clustering Algorithm model------------------------
# Original Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]

# New Data
new_costs = [1.2964818586720162, 1.667964983231757, 1.8405206274384565, 2.4454385483949457, 
             2.6289952648181285, 2.7037709795579286, 3.0528928800442934, 3.414389272794696, 
             4.019114145951091, 4.061721465751667]

# Combine Data
all_costs = costs + new_costs
x = np.arange(1, len(all_costs) + 1)

# Combine x and all costs into a single array
data = np.column_stack((x, all_costs))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_

# Function to find Pareto front
def pareto_frontier(Xs, Ys, maxX=True, maxY=False):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_front_X = [pair[0] for pair in p_front]
    p_front_Y = [pair[1] for pair in p_front]
    return p_front_X, p_front_Y

# Find Pareto front for combined data
p_front_X, p_front_Y = pareto_frontier(x, all_costs)

# Plot the scatter plot with clusters and Pareto front
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, all_costs, c=labels, cmap='viridis', label='Cost')
plt.colorbar(scatter, label='Cluster')
plt.plot(p_front_X, p_front_Y, color='red', linestyle='--', label='Pareto Front')

# Add value labels
for i, cost in enumerate(all_costs):
    plt.annotate(f'{cost:.2f}', (x[i], all_costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with K-Means Clustering and Pareto Front')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

#--------Second Clustering Algorithm model------------------------
# Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]
x = np.arange(1, len(costs) + 1)

# Combine x and costs into a single array
data = np.column_stack((x, costs))

# Apply DBSCAN clustering
db = DBSCAN(eps=3, min_samples=2).fit(data)
labels = db.labels_

# Plot the scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, costs, c=labels, cmap='viridis', label='Cost')
plt.colorbar(scatter, label='Cluster')
plt.plot(p_front_X, p_front_Y, color='red', linestyle='--', label='Pareto Front')

# Add value labels
for i, cost in enumerate(costs):
    plt.annotate(f'{cost:.2f}', (x[i], costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with DBSCAN Clustering')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

#-------------DBC Clustering Algorithm model------------------------
# Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]
x = np.arange(1, len(costs) + 1)

# Combine x and costs into a single array
data = np.column_stack((x, costs))

# Apply DBSCAN clustering
db = DBSCAN(eps=3, min_samples=2).fit(data)
labels = db.labels_

# Plot the scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, costs, c=labels, cmap='viridis', label='Cost')
plt.colorbar(scatter, label='Cluster')
plt.plot(p_front_X, p_front_Y, color='red', linestyle='--', label='Pareto Front')

# Add value labels
for i, cost in enumerate(costs):
    plt.annotate(f'{cost:.2f}', (x[i], costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with DBSCAN Clustering')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

#-------Hierarchial clustering----------------------
# Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]
x = np.arange(1, len(costs) + 1)

# Combine x and costs into a single array
data = np.column_stack((x, costs))

# Apply Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(data)

# Plot the scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, costs, c=labels, cmap='viridis', label='Cost')
plt.colorbar(scatter, label='Cluster')
plt.plot(p_front_X, p_front_Y, color='red', linestyle='--', label='Pareto Front')

# Add value labels
for i, cost in enumerate(costs):
    plt.annotate(f'{cost:.2f}', (x[i], costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with Agglomerative Clustering')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

#------ Gaussian mixture clustering ------------------
# Data
costs = [1.0884935606545056, 1.401030463573885, 2.2081470782222143, 2.859467331647874, 
         4.443448594557075, 5.294871846937422, 5.7420065440537345, 15.481036748293075, 
         19.77107999708505, 21.781522539832643]
x = np.arange(1, len(costs) + 1)

# Combine x and costs into a single array
data = np.column_stack((x, costs))

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(data)
labels = gmm.predict(data)

# Plot the scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, costs, c=labels, cmap='viridis', label='Cost')
plt.colorbar(scatter, label='Cluster')
plt.plot(p_front_X, p_front_Y, color='red', linestyle='--', label='Pareto Front')

# Add value labels
for i, cost in enumerate(costs):
    plt.annotate(f'{cost:.2f}', (x[i], costs[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Set titles and labels
plt.title('Scatter Plot of Costs for Top Nests with Gaussian Mixture Clustering')
plt.xlabel('Nest Index')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
