import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def generate_map():
    n = 26
    distances = [[random.randint(10, 100) for j in range(n)] for i in range(n)]
    with open("map.txt", "w") as f:
        f.write(str(n) + "\n")
        for i in range(n):
            f.write(" ".join(str(x) for x in distances[i]) + "\n")


def read_map():
    with open("map.txt", "r") as f:
        n = int(f.readline().strip())
        distances = []
        for i in range(n):
            row = [int(x) for x in f.readline().split()]
            distances.append(row)
    return n, distances


def print_map(n):
    print("Кількість міст: ", n)
    print("Матриця відстаней:")
    for i in range(n):
        print(Distances[i])


def ant_colony_optimization(distance_matrix, num_ants):
    n = len(distance_matrix)
    ant_locations = random.sample(range(n), num_ants)
    pheromone_matrix = [[PheromoneInitial for _ in range(n)] for _ in range(n)]
    shortest_distance = float('inf')
    shortest_path = []

    for iter in range(IterationNumber):
        for i in range(num_ants):
            current_location = ant_locations[i]
            allowed_locations = [j for j in range(n) if j != current_location]
            next_location = None

            if random.random() < q0:
                pheromone_values = [(j, pheromone_matrix[current_location][j]) for j in allowed_locations]
                pheromone_values.sort(key=lambda x: x[1], reverse=True)
                next_location = pheromone_values[0][0]
            else:
                probabilities = [(j, (pheromone_matrix[current_location][j]) ** alpha *
                                  (1 / distance_matrix[current_location][j]) ** beta) for j in allowed_locations]
                probabilities_sum = sum(p[1] for p in probabilities)
                probabilities = [(p[0], p[1] / probabilities_sum) for p in probabilities]
                probabilities.sort(key=lambda x: x[1], reverse=True)
                random_value = random.random()
                cumulative_probability = 0
                for p in probabilities:
                    cumulative_probability += p[1]
                    if random_value <= cumulative_probability:
                        next_location = p[0]
                        break

            ant_locations[i] = next_location

            for j in range(n):
                if j != current_location:
                    pheromone_matrix[current_location][j] *= (1 - EvaporationRate)
                    pheromone_matrix[current_location][j] += EvaporationRate * PheromoneInitial
                if j == next_location:
                    pheromone_matrix[current_location][j] += q0 * PheromoneInitial + (1 - q0) * \
                                                             pheromone_matrix[current_location][j]

        for i in range(n):
            for j in range(n):
                pheromone_matrix[i][j] *= (1 - EvaporationRate)

        for start in range(n):
            visited = {start}
            distance = 0
            path = [start]

            while len(visited) < n:
                current_location = path[-1]
                distances = [(j, distance_matrix[current_location][j]) for j in allowed_locations if j not in visited]
                if not distances:
                    break
                next_location = min(distances, key=lambda x: x[1])[0]
                visited.add(next_location)
                path.append(next_location)
                distance += distance_matrix[current_location][next_location]

            # add the distance back to the starting point
            distance += distance_matrix[path[-1]][start]

            if distance < shortest_distance:
                shortest_distance = distance
                shortest_path = path + [start]
        print(f"Iteration {iter + 1}: shortest path length = {shortest_distance}, shortest path = {shortest_path}")
    return shortest_path, shortest_distance


def plot_map(path, distance):
    path = path[:-1]
    n = len(path)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_coords = np.cos(theta)
    y_coords = np.sin(theta)
    plt.figure(figsize=(7, 6))
    plt.scatter(x_coords, y_coords, color='red', marker='o')
    plt.title('Cities')
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.box(True)
    plt.plot(x_coords, y_coords, color='green', alpha=1, linewidth=0.4)
    x_endpoint = np.array([x_coords[-1], x_coords[0]])
    y_endpoint = np.array([y_coords[-1], y_coords[0]])
    plt.plot(x_endpoint, y_endpoint, color='green', alpha=1, linewidth=0.4)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(path[i]) + '(' + str(i + 1) + ')', ha='center', va='bottom', fontsize=10)
    for i in range(n):
        for j in range(i + 1, n):
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], color='gray', alpha=0.1, linewidth=0.1)
    plt.show()


generate_map()
N, Distances = read_map()
alpha = 1
beta = 5
EvaporationRate = 0.5
q0 = 0.5
PheromoneInitial = 1.0
IterationNumber = 10
shortestPath, shortestDistance = ant_colony_optimization(Distances, N)
plot_map(shortestPath, N)
print(f"\nShortest path length = {shortestDistance}, shortest path = {shortestPath}")
