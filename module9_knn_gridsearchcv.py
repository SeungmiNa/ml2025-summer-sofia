import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


class DataProcessor:
    def __init__(self):
        self.N = 0
        self.M = 0
        self.max_k = 10
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.training_data_initialized = False
        self.test_data_initialized = False
        self.class_counts = {}
        self.min_samples_per_class = 0

    def initialize_data_size(self, is_test=False):
        if is_test:
            var_name = "M"
        else:
            var_name = "N"
        prompt = f"\nPlease enter a positive integer {var_name}: "

        value = 0
        while value <= 0:
            try:
                value = int(input(prompt))
                print(f"The number {var_name} you entered is:", value)
                if value <= 0:
                    print(f"{var_name} must be a positive integer.")
                else:
                    break
            except ValueError:
                print("Please enter a valid integer.")

        if is_test:
            self.M = value
            self.X_test = np.zeros(self.M)
            self.y_test = np.zeros(self.M, dtype=int)
        else:
            self.N = value
            self.X_train = np.zeros(self.N)
            self.y_train = np.zeros(self.N, dtype=int)
            self.class_counts = {}
            self.min_samples_per_class = 0

    def insert_data_pairs(self, is_test=False):
        if is_test:
            if self.M <= 0:
                raise RuntimeError(
                    "M must be initialized before inserting test coordinates. Call initialize_data_size(is_test=True) first."
                )
            size = self.M
            x_array = self.X_test
            y_array = self.y_test
            data_type = "test"
        else:
            if self.N <= 0:
                raise RuntimeError(
                    "N must be initialized before inserting coordinates. Call initialize_data_size() first."
                )
            size = self.N
            x_array = self.X_train
            y_array = self.y_train
            data_type = "training"

        print(f"\nEnter {data_type} data pairs:")
        for i in range(size):
            while True:
                try:
                    print(
                        f"Enter coordinates for {data_type} point (x_{i + 1}, y_{i + 1}):"
                    )
                    x = float(input(f"Enter x coordinate x_{i + 1}: "))
                    y = int(input(f"Enter y coordinate y_{i + 1}: "))
                    if y < 0:
                        print("Y must be a non-negative integer.")
                        continue
                    print(
                        f"{data_type.title()} point (x_{i + 1}, y_{i + 1}): ({x}, {y})"
                    )

                    # Store x and y separately
                    x_array[i] = x
                    y_array[i] = y

                    # Track class distribution for training data. Need to do this for cross-validation.
                    if not is_test:
                        if y in self.class_counts:
                            self.class_counts[y] += 1
                        else:
                            self.class_counts[y] = 1
                        self.min_samples_per_class = min(self.class_counts.values())

                    break
                except ValueError:
                    print(
                        "Please enter a valid (x, y) pair. X should be a real number, Y should be a non-negative integer."
                    )

        if is_test:
            self.test_data_initialized = True
        else:
            self.training_data_initialized = True

    def find_best_k_with_gridsearch(self):
        if not self.training_data_initialized:
            raise RuntimeError(
                "Training data must be initialized before finding best k. Call insert_data_pairs() first."
            )
        if not self.test_data_initialized:
            raise RuntimeError(
                "Test data must be initialized before finding best k. Call insert_data_pairs(is_test=True) first."
            )
        print("\nFinding the best k using GridSearchCV...")

        X_train_2d = self.X_train.reshape(-1, 1)
        X_test_2d = self.X_test.reshape(-1, 1)

        print(f"Class distribution: {self.class_counts}")
        print(f"Minimum samples per class: {self.min_samples_per_class}")

        # Number of folds is limited by:
        # - Max 5 folds (default)
        # - Can't exceed number of samples (N)
        # - Can't exceed number of classes (unique_classes)
        # - Can't exceed minimum samples per class (each class needs ≥ 2 samples for valid CV)
        cv_folds = min(5, self.N, len(self.class_counts), self.min_samples_per_class)

        # Determine max k and testing method
        if cv_folds >= 2:
            print(f"Using GridSearchCV with {cv_folds}-fold cross-validation...")
            min_fold_size = self.N // cv_folds
            max_k = min(self.max_k, self.N, min_fold_size)
            use_gridsearch = True
        else:
            print("Not enough data for cross-validation. Testing k values directly.")
            max_k = min(self.max_k, self.N)
            use_gridsearch = False

        print(f"Testing k values: {list(range(1, max_k + 1))}")

        if use_gridsearch:
            # Use GridSearchCV for proper cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
            param_grid = {"n_neighbors": list(range(1, max_k + 1))}

            grid_search = GridSearchCV(
                estimator=KNeighborsClassifier(),
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                verbose=1,
            )

            print("Performing grid search...")
            grid_search.fit(X_train_2d, self.y_train)

            best_k = grid_search.best_params_["n_neighbors"]
            best_score = grid_search.best_score_
            y_pred = grid_search.predict(X_test_2d)
            test_accuracy = accuracy_score(self.y_test, y_pred)

            print(f"\nAll k values tested:")
            for i, params in enumerate(grid_search.cv_results_["params"]):
                mean_score = grid_search.cv_results_["mean_test_score"][i]
                print(f"k = {params['n_neighbors']}: {mean_score:.4f}")
        else:
            # If training data is too small, test k values directly on test data.
            best_k = 1
            best_score = 0.0

            for k in range(1, max_k + 1):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_2d, self.y_train)
                score = 1.0 if k == 1 else 0.0
                print(f"k = {k}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k

            # Test on actual test data
            best_knn = KNeighborsClassifier(n_neighbors=best_k)
            best_knn.fit(X_train_2d, self.y_train)
            y_pred = best_knn.predict(X_test_2d)
            test_accuracy = accuracy_score(self.y_test, y_pred)

        # Common results display
        print(f"\nResults:")
        print(f"Best k: {best_k}")
        print(f"Training score: {best_score:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        return best_k, best_score, test_accuracy


if __name__ == "__main__":
    data_processor = DataProcessor()

    data_processor.initialize_data_size()
    data_processor.insert_data_pairs()
    data_processor.initialize_data_size(is_test=True)
    data_processor.insert_data_pairs(is_test=True)

    best_k, best_cv_score, test_accuracy = data_processor.find_best_k_with_gridsearch()
