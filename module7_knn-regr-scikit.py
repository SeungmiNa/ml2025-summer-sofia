import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class DataProcessor:
    def __init__(self):
        self.N = 0
        self.k = 0
        self.X_train = None
        self.y_train = None
        self.X = None
        self.data_initialized = False
        self.knn_model = None
    
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
        
        self.X_train = np.zeros(self.N)
        self.y_train = np.zeros(self.N)

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
                    
                    # Store x and y separately
                    self.X_train[i] = x
                    self.y_train[i] = y
                    break
                except ValueError:
                    print("Please enter a valid (x, y) pair.")
                    # Continue in the same iteration
        self.data_initialized = True
        self._initialize_knn_model()

    def _initialize_knn_model(self):
        if self.data_initialized and self.k > 0:
            # Reshape X_train for sklearn (2D array)
            X_train_2d = self.X_train.reshape(-1, 1)
            
            # Create and fit the KNN model
            self.knn_model = KNeighborsRegressor(n_neighbors=self.k, algorithm='auto')
            self.knn_model.fit(X_train_2d, self.y_train)
            print(f"KNN model initialized with k = {self.k}")

    def knn_regression(self):
        if self.N <= 0:
            raise RuntimeError("N must be initialized before performing KNN regression. Call initialize_N() first.")
        if self.k <= 0:
            raise RuntimeError("k must be initialized before performing KNN regression. Call initialize_k() first.")
        if not self.data_initialized:
            raise RuntimeError("Data must be initialized before performing KNN regression. Call insert_x_y_pairs() first.")
        if self.knn_model is None:
            raise RuntimeError("KNN model not initialized. Please ensure data is properly set.")

        while not self.X:
            try:
                self.X = float(input("Please enter a number X: "))
                print("The number you entered is:", self.X)
            except ValueError:
                print("Please enter a valid number.")
        
        print("Performing KNN regression using scikit-learn...")

        if self.k > self.N:
            raise RuntimeError(f"k ({self.k}) cannot be greater than N ({self.N}). Not enough data points for k-nearest neighbors.")
        
        X_pred = np.array([[self.X]])  # Reshape for sklearn (2D array)
        predicted_y = self.knn_model.predict(X_pred)[0]
        
        # Get the k nearest neighbors using sklearn's kneighbors method
        distances, indices = self.knn_model.kneighbors(X_pred)
        
        print(f"K nearest neighbors (k = {self.k}):")
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            x_val = self.X_train[idx]
            y_val = self.y_train[idx]
            print(f"* Neighbor {i + 1}: Point {idx + 1} ({x_val:.2f}, {y_val:.2f}) - Distance: {distance:.4f}")
        
        # Additionally, calculate and display variance of training labels
        y_variance = np.var(self.y_train)
        print(f"Predicted Y for X = {self.X}: {predicted_y:.4f}")
        print(f"Variance of training labels: {y_variance:.4f}")
        
        return predicted_y
                
data_processor = DataProcessor()
data_processor.initialize_N()
data_processor.initialize_k()
data_processor.insert_x_y_pairs()
predicted_y = data_processor.knn_regression()