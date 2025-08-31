import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix


class DataProcessor:
    def __init__(self):
        self.N = 0
        self.x = None
        self.y = None
        self.data_initialized = False

    def initialize_N(self):
        while self.N <= 0:
            try:
                self.N = int(input("Please enter a positive integer N: "))
                if self.N <= 0:
                    print("N must be a positive integer.")
                else:
                    print(f"We will collect {self.N} points.")
                    self.x = np.zeros(self.N)
                    self.y = np.zeros(self.N)
                    break
            except ValueError:
                print("Please enter a valid integer.")

    def initialize_data(self):
        if self.N <= 0:
            raise RuntimeError(
                "N must be initialized before inserting class labels (x) and predictions (y). Call initialize_N() first."
            )

        print("Enter class labels and predictions for each point (0 or 1):")

        for i in range(self.N):
            print(f"Point {i + 1}:")
            while True:
                x_input = input("→ Enter label: ").strip()
                y_input = input("→ Enter prediction: ").strip()

                # Need to check if x and y are exactly 0 or 1
                # Instead of using try-except to validate the input is an integer first,
                # we can check if the input is exactly "0" or "1".
                # If it is, we can convert it to an integer directly.
                if x_input in ["0", "1"] and y_input in ["0", "1"]:
                    self.x[i] = int(x_input)
                    self.y[i] = int(y_input)
                    break
                print("Both x and y must be exactly 0 or 1. Please try again.")

        self.data_initialized = True

        print(f"\n=== Data Summary ===")
        for i in range(self.N):
            label, pred = int(self.x[i]), int(self.y[i])
            if label == 1 and pred == 1:
                result = "TP"
            elif label == 0 and pred == 0:
                result = "TN"
            elif label == 0 and pred == 1:
                result = "FP"
            else:
                result = "FN"
            print(f"Point {i + 1}: label = {label}, prediction = {pred} → {result}")

        cm = confusion_matrix(self.x, self.y, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"Total # of points: {tp + tn + fp + fn}")

    def calculate_metrics(self):
        """Calculate precision and recall using scikit-learn and print results"""
        if self.N <= 0 or not self.data_initialized:
            raise RuntimeError(
                "Data must be initialized and inserted before calculating metrics."
            )

        precision = precision_score(self.x, self.y, zero_division=0)
        recall = recall_score(self.x, self.y, zero_division=0)

        print("\n=== Results ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return precision, recall


if __name__ == "__main__":
    processor = DataProcessor()
    processor.initialize_N()
    processor.initialize_data()
    processor.calculate_metrics()
