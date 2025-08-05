class DataProcessor:
    def __init__(self):
        self.N = 0
        self.numbers = []
        self.X = None
        self.is_initialized = False
        self.is_inserted = False
    
    def initialize_data(self):
        while self.N <= 0:
            try:
                self.N = int(input("Please enter a positive integer N: "))
                print("The number you entered is:", self.N)
                if self.N <= 0:
                    print("N must be a positive integer.")
                else:
                    break
            except ValueError:
                print("Please enter a valid integer.")
        self.is_initialized = True
                
    def insert_data(self):
        if not self.is_initialized:
            print("Please initialize the data first.")
            return
        i = 0
        while i < self.N:
            try:
                number = int(input(f"Enter number {i + 1}: "))
                print(f"The number {i + 1} you entered is:", number)
                self.numbers.append(number)
                i += 1
            except ValueError:
                print("Please enter a valid integer.")
                # Stay in the same iteration by not incrementing i
        self.is_inserted = True

    def search_data(self):
        if not self.is_inserted:
            print("Please insert the data first.")
            return
        while self.X is None:
            try:
                self.X = int(input("Please enter a number X to search for: "))
            except ValueError:
                print("Please enter a valid integer.")

        for i in range(self.N):
            if self.numbers[i] == self.X:
                print(i + 1)
                break
        else:
            print(-1)