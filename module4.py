# The program asks the user for input N (positive integer) and reads it
N = 0
while N <= 0:
    try:
        N = int(input("Please enter a positive integer N: "))
        print("The number you entered is:", N)
        if N <= 0:
            print("N must be a positive integer.")
        else:
            break
    except ValueError:
        print("Please enter a valid integer.")

# Then the program asks the user to provide N numbers (one by one) and reads all of them (again, one by one)
print(f"Please enter {N} numbers one by one:")
numbers = []
i = 0
while i < N:
    try:
        number = int(input(f"Enter number {i + 1}: "))
        print(f"The number {i + 1} you entered is:", number)
        numbers.append(number)
        i += 1
    except ValueError:
        print("Please enter a valid integer.")
        # Stay in the same iteration by not incrementing i

# In the end, the program asks the user for input X (integer) and outputs: "-1" if there were no such X among N read numbers, or the index (from 1 to N) of this X if the user inputed it before.
X = None
while X is None:
    try:
        X = int(input("Please enter a number X to search for: "))
    except ValueError:
        print("Please enter a valid integer.")

for i in range(N):
    if numbers[i] == X:
        print(i + 1)
        break
else:
    print(-1)
