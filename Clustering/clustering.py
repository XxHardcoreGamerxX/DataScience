import sys
import pandas as pd
import math
"""
File reader function that utilises the pandas library.

Parameters:
- inputfile (str): Path to the input file to be read

Returns:
- DataFrame: A pandas DataFrame containing the data from the file with columns 'object_id', 'x', and 'y'.
"""
def file_reader(inputfile):
    df = pd.read_csv(inputfile, sep='\t', header=None, names=['object_id','x', 'y'])
    return df
"""
Function that calculates the euclidean distance

Parameters:
- p1 (list or tuple): The first point as a list or tuple with at least two elements representing coordinates.
- p2 (list or tuple): The second point as a list or tuple with at least two elements representing coordinates.
"""
def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

"""
A function that will use the distance and epsilon to gather the neighbors around a point

Parameters:
- point (list or tuple): The reference point to find neighbors for.
- data (list): A list of points where each point is a list or tuple containing coordinates.
- eps (float): The maximum distance to consider a point as a neighbor.
"""
def get_neighbours(point, data, eps):
    neighbours = []
    for other_point in data:
        if distance(point, other_point) <= eps:
            neighbours.append(other_point)
    return neighbours

"""
Function that will use get neighbours in order to add points to the cluster if they are in the neighbourhood of a given point

Parameters:
- point (list or tuple): The reference point for clustering.
- neighbours (list): The list of neighbors for the reference point.
- clusters (dict): A dictionary mapping point IDs to cluster IDs.
- cluster_id (int): The ID of the current cluster.
- data (list): A list of all points.
- eps (float): The maximum distance to consider a point as a neighbor.
- min_pts (int): The minimum number of points required to form a dense region.
"""
def addtocluster(point, neighbours, clusters, cluster_id, data, eps, min_pts):
    clusters[point[0]] = cluster_id
    i = 0
    while i < len(neighbours):
        neighbour = neighbours[i]
        if neighbour[0] not in clusters:
            clusters[neighbour[0]] = cluster_id
            new_neighbour = get_neighbours(neighbour, data, eps)
            if len(new_neighbour) >= min_pts:
                neighbours += new_neighbour
        i += 1
 
"""
DBSCAN clustering algorithm implementation.

Parameters:
- data (list): A list of points where each point is a list or tuple containing coordinates.
- eps (float): The maximum distance to consider a point as a neighbor.
- min_pts (int): The minimum number of points required to form a dense region.
"""
def dbScan(data, eps, min_pts):
    clusters = {}
    cluster_id = 0
    for point in data:
        if point[0] not in clusters:
            neighbours = get_neighbours(point, data, eps)
            if len(neighbours) >= min_pts:
                addtocluster(point, neighbours, clusters, cluster_id, data, eps, min_pts)
                cluster_id += 1
            else:
                clusters[point[0]] = -1
    return clusters

"""
Function to write clusters to output files.

Parameters:
- clusters (dict): A dictionary mapping point IDs to cluster IDs.
- n (int): The number of largest clusters to write to files.
- filename (str): The base name for the output files.
"""
def outfilewrite(clusters, n, filename):
    cluster_files = {}
    for point_id, cluster_id in clusters.items():
        if cluster_id not in cluster_files:
            cluster_files[cluster_id] = []
        cluster_files[cluster_id].append(int(point_id))

    cluster_files = sorted(cluster_files.items(), key=lambda x: len(x[1]), reverse=True)
    for i in range(min(n, len(cluster_files))):
        output_filename = f"{filename}_cluster_{i}.txt"
        with open(output_filename, 'w') as file:
            for point_id in cluster_files[i][1]:
                file.write(f"{point_id}\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python studentID_name_hw3.py <input_file> <n> <Eps> <MinPts>")
        sys.exit(1)

    input_file = sys.argv[1]
    n = int(sys.argv[2])
    eps = float(sys.argv[3])
    min_pts = int(sys.argv[4])

    df = file_reader(input_file)
    data = df.values.tolist()
    clusters = dbScan(data, eps, min_pts)
    outfilewrite(clusters, n, input_file.split('.')[0])
