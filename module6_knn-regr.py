import numpy as np

class DataProcessor:
    def __init__(self):
        self.N = 0
        self.k = 0
        self.coordinates = None
        self.X = None
        self.coordinates_initialized = False
    
    def initialize_N(self):
        while self.N <= 0:
            try:
                self.N = int(input("Please enter a positive integer N: "))
                print("The number N you entered is:", self.N)
                if self.N <= 0:
                    print("N must be a positive integer.")
                else:
                    break
            except ValueError:
                print("Please enter a valid integer.")
        
        self.coordinates = np.zeros((self.N, 2))

    def initialize_k(self):
        if self.N <= 0:
            raise RuntimeError("N must be initialized before k. Call initialize_N() first.")
        
        while self.k <= 0:
            try:
                self.k = int(input("Please enter a positive integer k: "))
                print("The number k you entered is:", self.k)
                if self.k <= 0:
                    print("k must be a positive integer.")
                else:
                    break
            except ValueError:
                print("Please enter a valid integer.")
                
    def insert_x_y_pairs(self):
        if self.N <= 0:
            raise RuntimeError("N must be initialized before inserting coordinates. Call initialize_N() first.")
        if self.k <= 0:
            raise RuntimeError("k must be initialized before inserting coordinates. Call initialize_k() first.")
        
        for i in range(self.N):
            while True:
                try:
                    print(f"Enter coordinates for point (x_{i + 1}, y_{i + 1}):")
                    x = float(input(f"Enter x coordinate x_{i + 1}: "))
                    y = float(input(f"Enter y coordinate y_{i + 1}: "))
                    print(f"Point (x_{i + 1}, y_{i + 1}): ({x}, {y})")
                    
                    # Fill the numpy array directly
                    self.coordinates[i, 0] = x
                    self.coordinates[i, 1] = y
                    break
                except ValueError:
                    print("Please enter a valid (x, y) pair.")
                    # Continue in the same iteration
        self.coordinates_initialized = True

    def knn_regression(self):
        if self.N <= 0:
            raise RuntimeError("N must be initialized before performing KNN regression. Call initialize_N() first.")
        if self.k <= 0:
            raise RuntimeError("k must be initialized before performing KNN regression. Call initialize_k() first.")
        if not self.coordinates_initialized:
            raise RuntimeError("Coordinates must be initialized before performing KNN regression. Call insert_x_y_pairs() first.")

        while not self.X:
            try:
                self.X = float(input("Please enter a number X: "))
                print("The number you entered is:", self.X)
            except ValueError:
                print("Please enter a valid number.")
        
        print("Performing KNN regression...")

        if self.k > self.N:
            raise RuntimeError(f"k ({self.k}) cannot be greater than N ({self.N}). Not enough data points for k-nearest neighbors.")
        
        # Extract x and y coordinates from the numpy array
        x_coords = self.coordinates[:, 0]
        y_coords = self.coordinates[:, 1]
        # Calculate distances from X to all x-coordinates using numpy
        distances = np.abs(self.X - x_coords)
        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        # Get the k nearest neighbors
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_y = y_coords[k_nearest_indices]
        k_nearest_coords = self.coordinates[k_nearest_indices]
        # Calculate the average y-value of k nearest neighbors
        predicted_y = np.mean(k_nearest_y)
        
        print(f"K nearest neighbors (k = {self.k}):")
        for i, (idx, distance, y_val, coords) in enumerate(zip(k_nearest_indices, k_nearest_distances, k_nearest_y, k_nearest_coords)):
            print(f"* Neighbor {i + 1}: Point {idx + 1} ({coords[0]:.2f}, {coords[1]:.2f}) - Distance: {distance:.4f}")
        
        print(f"Predicted Y for X = {self.X}: {predicted_y:.4f}")
        
        return predicted_y
                
data_processor = DataProcessor()
data_processor.initialize_N()
data_processor.initialize_k()
data_processor.insert_x_y_pairs()
predicted_y = data_processor.knn_regression()