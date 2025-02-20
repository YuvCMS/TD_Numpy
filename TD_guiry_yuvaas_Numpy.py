import numpy as np

import time

# Exercise 1: Creating and Manipulating NumPy Arrays

# 1. Create a 1D NumPy array from the list [5, 10, 15, 20, 25]. Convert the array to type float64 and print it.
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("1D Array:", array_1d)

# 2. Create a 2D NumPy array from the nested list [[1, 2, 3], [4, 5, 6], [7, 8, 9]]. Print the shape and size of the array.
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array Shape:", array_2d.shape)
print("2D Array Size:", array_2d.size)

# 3. Create a 3D NumPy array with random values of shape (2, 3, 4). Print the number of dimensions and the shape of the array.
array_3d = np.random.rand(2, 3, 4)
print("3D Array Dimensions:", array_3d.ndim)
print("3D Array Shape:", array_3d.shape)


# Exercise 2: Advanced Array Manipulations

# 1. Create a 1D NumPy array with the numbers from 0 to 9. Reverse the array and print it.
array_1d_rev = np.arange(10)[::-1]
print("Reversed 1D Array:", array_1d_rev)

# 2. Create a 2D NumPy array with the numbers from 0 to 11, arranged in a 3x4 shape. 
# Extract a subarray consisting of the first two rows and the last two columns, and print it.
array_2d_sub = np.arange(12).reshape(3, 4)
subarray = array_2d_sub[:2, -2:]
print("Original 3x4 Array:\n", array_2d_sub)
print("Extracted Subarray:\n", subarray)

# 3. Create a 2D NumPy array of shape (5, 5) with random integers between 0 and 10. 
# Replace all elements greater than 5 with 0 and print the modified array.
array_2d_rand = np.random.randint(0, 11, (5, 5))
array_2d_rand[array_2d_rand > 5] = 0
print("Modified 5x5 Array:\n", array_2d_rand)


# Exercise 3: Array Initialization and Attributes

# 1. Create a 3x3 identity matrix using NumPy and print its attributes: ndim, shape, size, itemsize, and nbytes.
identity_matrix = np.eye(3)
print("Identity Matrix:\n", identity_matrix)
print("Attributes: ndim =", identity_matrix.ndim, ", shape =", identity_matrix.shape, 
      ", size =", identity_matrix.size, ", itemsize =", identity_matrix.itemsize, 
      ", nbytes =", identity_matrix.nbytes)

# 2. Create an array of 10 evenly spaced numbers between 0 and 5 using numpy.linspace(). Print the array and its datatype.
linspace_array = np.linspace(0, 5, 10)
print("Linspace Array:", linspace_array)
print("Data Type:", linspace_array.dtype)

# 3. Create a 3D array of shape (2, 3, 4) with random values from a standard normal distribution. 
# Print the array and the sum of all elements.
array_3d_normal = np.random.randn(2, 3, 4)
print("3D Normal Distribution Array:\n", array_3d_normal)
print("Sum of Elements:", array_3d_normal.sum())



# Exercise 4: Fancy Indexing and Masking

# 1. Create a 1D NumPy array with random integers between 0 and 50 of size 20.
array_1d = np.random.randint(0, 50, 20)
indices = [2, 5, 7, 10, 15]
selected_elements = array_1d[indices]
print("Original 1D Array:", array_1d)
print("Selected Elements:", selected_elements)

# 2. Create a 2D NumPy array with random integers between 0 and 30 of shape (4, 5).
array_2d = np.random.randint(0, 30, (4, 5))
boolean_mask = array_2d > 15
selected_elements_mask = array_2d[boolean_mask]
print("Original 2D Array:\n", array_2d)
print("Elements > 15:\n", selected_elements_mask)

# 3. Create a 1D NumPy array of 10 random integers between -10 and 10.
array_1d_neg = np.random.randint(-10, 10, 10)
array_1d_neg[array_1d_neg < 0] = 0
print("Modified 1D Array (Negative values set to 0):", array_1d_neg)


# Exercise 5: Combining and Splitting Arrays

# 1. Create two 1D NumPy arrays of length 5 with random integers between 0 and 10. Concatenate the two arrays.
array_a = np.random.randint(0, 10, 5)
array_b = np.random.randint(0, 10, 5)
concatenated_array = np.concatenate((array_a, array_b))
print("Array A:", array_a)
print("Array B:", array_b)
print("Concatenated Array:", concatenated_array)

# 2. Create a 2D NumPy array of shape (6, 4) with random integers between 0 and 10. Split it into two equal parts along the row axis.
array_2d_split = np.random.randint(0, 10, (6, 4))
split_arrays = np.split(array_2d_split, 2, axis=0)
print("Original 6x4 Array:\n", array_2d_split)
print("Split Arrays along rows:\n", split_arrays[0], "\n", split_arrays[1])

# 3. Create a 2D NumPy array of shape (3, 6) with random integers between 0 and 10. Split it into three equal parts along the column axis.
array_2d_col_split = np.random.randint(0, 10, (3, 6))
split_arrays_col = np.split(array_2d_col_split, 3, axis=1)
print("Original 3x6 Array:\n", array_2d_col_split)
print("Split Arrays along columns:\n", split_arrays_col[0], "\n", split_arrays_col[1], "\n", split_arrays_col[2])


# Exercise 6: Mathematical Functions and Aggregations

# 1. Create a 1D NumPy array with random integers between 1 and 100 of size 15. Compute and print mean, median, standard deviation, and variance.
array_stats = np.random.randint(1, 100, 15)
print("Array for Statistics:", array_stats)
print("Mean:", np.mean(array_stats))
print("Median:", np.median(array_stats))
print("Standard Deviation:", np.std(array_stats))
print("Variance:", np.var(array_stats))

# 2. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 50. Compute and print the sum of each row and each column.
array_2d_sum = np.random.randint(1, 50, (4, 4))
row_sums = np.sum(array_2d_sum, axis=1)
col_sums = np.sum(array_2d_sum, axis=0)
print("Original 4x4 Array:\n", array_2d_sum)
print("Row Sums:", row_sums)
print("Column Sums:", col_sums)

# 3. Create a 3D NumPy array of shape (2, 3, 4) with random integers between 1 and 20. Find max and min values along each axis.
array_3d = np.random.randint(1, 20, (2, 3, 4))
max_values = np.max(array_3d, axis=0)
min_values = np.min(array_3d, axis=0)
print("3D Array:\n", array_3d)
print("Max Values along Axis 0:\n", max_values)
print("Min Values along Axis 0:\n", min_values)



# Exercise 7: Reshaping and Transposing Arrays

# 1. Create a 1D NumPy array with the numbers from 1 to 12. Reshape it to a 2D array of shape (3, 4) and print it.
array_1d = np.arange(1, 13)
array_reshaped = array_1d.reshape(3, 4)
print("Reshaped (3,4) Array:\n", array_reshaped)

# 2. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10. Transpose the array and print it.
array_2d = np.random.randint(1, 10, (3, 4))
transposed_array = array_2d.T
print("Original (3,4) Array:\n", array_2d)
print("Transposed Array:\n", transposed_array)

# 3. Create a 2D NumPy array of shape (2, 3) with random integers between 1 and 10. Flatten the array and print it.
array_2d_flat = np.random.randint(1, 10, (2, 3))
flattened_array = array_2d_flat.flatten()
print("Original (2,3) Array:\n", array_2d_flat)
print("Flattened Array:", flattened_array)


# Exercise 8: Broadcasting and Vectorized Operations

# 1. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10.
# Subtract the mean of each column from the respective column elements and print the result.
array_2d_broadcast = np.random.randint(1, 10, (3, 4))
mean_columns = array_2d_broadcast.mean(axis=0)
normalized_array = array_2d_broadcast - mean_columns
print("Original Array:\n", array_2d_broadcast)
print("Mean Subtracted Array:\n", normalized_array)

# 2. Create two 1D NumPy arrays of length 4 with random integers between 1 and 5.
# Use broadcasting to compute and print the outer product of the two arrays.
array_1 = np.random.randint(1, 5, 4)
array_2 = np.random.randint(1, 5, 4)
outer_product = np.outer(array_1, array_2)
print("Array 1:", array_1)
print("Array 2:", array_2)
print("Outer Product:\n", outer_product)

# 3. Create a 2D NumPy array of shape (4, 5) with random integers between 1 and 10.
# Add 10 to all elements greater than 5 and print the modified array.
array_2d_modify = np.random.randint(1, 10, (4, 5))
array_2d_modify[array_2d_modify > 5] += 10
print("Modified Array:\n", array_2d_modify)


# Exercise 9: Sorting and Searching Arrays

# 1. Create a 1D NumPy array with random integers between 1 and 20 of size 10.
# Sort the array in ascending order and print it.
array_sort_1d = np.random.randint(1, 20, 10)
sorted_array = np.sort(array_sort_1d)
print("Original 1D Array:", array_sort_1d)
print("Sorted Array:", sorted_array)

# 2. Create a 2D NumPy array of shape (3, 5) with random integers between 1 and 50.
# Sort the array by the second column and print the result.
array_2d_sort = np.random.randint(1, 50, (3, 5))
sorted_by_column = array_2d_sort[array_2d_sort[:, 1].argsort()]
print("Original (3,5) Array:\n", array_2d_sort)
print("Array Sorted by Second Column:\n", sorted_by_column)

# 3. Create a 1D NumPy array with random integers between 1 and 100 of size 15.
# Find and print the indices of all elements greater than 50.
array_search = np.random.randint(1, 100, 15)
indices = np.where(array_search > 50)
print("Original Array:", array_search)
print("Indices of Elements Greater than 50:", indices[0])


# Exercise 10: Linear Algebra with NumPy

# 1. Create a 2D NumPy array of shape (2, 2) with random integers between 1 and 10.
# Compute and print the determinant of the array.
array_2x2 = np.random.randint(1, 10, (2, 2))
determinant = np.linalg.det(array_2x2)
print("Matrix (2x2):\n", array_2x2)
print("Determinant:", determinant)

# 2. Create a 2D NumPy array of shape (3, 3) with random integers between 1 and 5.
# Compute and print the eigenvalues and eigenvectors of the array.
array_3x3 = np.random.randint(1, 5, (3, 3))
eigenvalues, eigenvectors = np.linalg.eig(array_3x3)
print("Matrix (3x3):\n", array_3x3)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 3. Create two 2D NumPy arrays of shape (2, 3) and (3, 2) with random integers between 1 and 10.
# Compute and print the matrix product of the two arrays.
matrix_A = np.random.randint(1, 10, (2, 3))
matrix_B = np.random.randint(1, 10, (3, 2))
matrix_product = np.dot(matrix_A, matrix_B)
print("Matrix A (2x3):\n", matrix_A)
print("Matrix B (3x2):\n", matrix_B)
print("Matrix Product (2x2):\n", matrix_product)


# Exercise 11: Random Sampling and Distributions

# 1. Create a 1D NumPy array of 10 random samples from a uniform distribution over [0, 1] and print the array.
array_uniform = np.random.uniform(0, 1, 10)
print("Uniform Distribution Array:", array_uniform)

# 2. Create a 2D NumPy array of shape (3, 3) with random samples from a normal distribution 
# with mean 0 and standard deviation 1. Print the array.
array_normal = np.random.normal(0, 1, (3, 3))
print("Normal Distribution (3x3) Array:\n", array_normal)

# 3. Create a 1D NumPy array of 20 random integers between 1 and 100. 
# Compute and print the histogram of the array with 5 bins.
array_hist = np.random.randint(1, 100, 20)
hist, bins = np.histogram(array_hist, bins=5)
print("Random Integer Array:", array_hist)
print("Histogram Bins:", bins)
print("Histogram Counts:", hist)


# Exercise 12: Advanced Indexing and Selection

# 1. Create a 2D NumPy array of shape (5, 5) with random integers between 1 and 20.
# Select and print the diagonal elements of the array.
array_diag = np.random.randint(1, 20, (5, 5))
diagonal_elements = np.diag(array_diag)
print("Original (5x5) Array:\n", array_diag)
print("Diagonal Elements:", diagonal_elements)

# 2. Create a 1D NumPy array of 10 random integers between 1 and 50.
# Use advanced indexing to select and print all elements that are prime numbers.
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

array_primes = np.random.randint(1, 50, 10)
prime_mask = np.vectorize(is_prime)(array_primes)
primes = array_primes[prime_mask]
print("Original Array:", array_primes)
print("Prime Numbers:", primes)

# 3. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 10.
# Select and print all elements that are even numbers.
array_even = np.random.randint(1, 10, (4, 4))
even_numbers = array_even[array_even % 2 == 0]
print("Original (4x4) Array:\n", array_even)
print("Even Numbers:", even_numbers)


# Exercise 13: Handling Missing Data

# 1. Create a 1D NumPy array of length 10 with random integers between 1 and 10.
# Introduce np.nan at random positions and print the array.
array_missing = np.random.randint(1, 10, 10).astype(float)
nan_positions = np.random.choice(10, 3, replace=False)  # Randomly choose 3 positions for NaN
array_missing[nan_positions] = np.nan
print("Array with Missing Values:", array_missing)

# 2. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10.
# Replace all elements that are less than 5 with np.nan and print the array.
array_2d_nan = np.random.randint(1, 10, (3, 4)).astype(float)
array_2d_nan[array_2d_nan < 5] = np.nan
print("Modified (3x4) Array with np.nan:\n", array_2d_nan)

# 3. Create a 1D NumPy array of length 15 with random integers between 1 and 20.
# Identify and print the indices of all np.nan values in the array.
array_nan_search = np.random.randint(1, 20, 15).astype(float)
nan_positions = np.random.choice(15, 4, replace=False)  # Randomly set 4 NaNs
array_nan_search[nan_positions] = np.nan
nan_indices = np.where(np.isnan(array_nan_search))
print("Array with NaNs:", array_nan_search)
print("Indices of np.nan values:", nan_indices[0])



# Exercise 14: Performance Optimization with NumPy

# 1. Create a large 1D NumPy array with 1 million random integers between 1 and 100.
# Compute the mean and standard deviation using NumPy functions and measure the time taken.
large_array = np.random.randint(1, 100, 1_000_000)
start_time = time.time()
mean_value = np.mean(large_array)
std_value = np.std(large_array)
end_time = time.time()
print(f"Mean: {mean_value}, Standard Deviation: {std_value}, Time Taken: {end_time - start_time} seconds")

# 2. Create two large 2D NumPy arrays of shape (1000, 1000) with random integers between 1 and 10.
# Perform element-wise addition and measure the time taken.
array_A = np.random.randint(1, 10, (1000, 1000))
array_B = np.random.randint(1, 10, (1000, 1000))
start_time = time.time()
result = array_A + array_B
end_time = time.time()
print(f"Element-wise addition time: {end_time - start_time} seconds")

# 3. Create a 3D NumPy array of shape (100, 100, 100) with random integers between 1 and 10.
# Compute the sum along each axis and measure the time taken.
array_3d = np.random.randint(1, 10, (100, 100, 100))
start_time = time.time()
sum_axis0 = np.sum(array_3d, axis=0)
sum_axis1 = np.sum(array_3d, axis=1)
sum_axis2 = np.sum(array_3d, axis=2)
end_time = time.time()
print(f"Summation time: {end_time - start_time} seconds")


# Exercise 15: Cumulative and Aggregate Functions

# 1. Create a 1D NumPy array with the numbers from 1 to 10.
# Compute and print the cumulative sum and cumulative product of the array.
array_cum = np.arange(1, 11)
cum_sum = np.cumsum(array_cum)
cum_prod = np.cumprod(array_cum)
print("Cumulative Sum:", cum_sum)
print("Cumulative Product:", cum_prod)

# 2. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 20.
# Compute and print the cumulative sum along the rows and the columns.
array_2d_cum = np.random.randint(1, 20, (4, 4))
cum_sum_rows = np.cumsum(array_2d_cum, axis=1)
cum_sum_cols = np.cumsum(array_2d_cum, axis=0)
print("Original (4x4) Array:\n", array_2d_cum)
print("Cumulative Sum Along Rows:\n", cum_sum_rows)
print("Cumulative Sum Along Columns:\n", cum_sum_cols)

# 3. Create a 1D NumPy array with 10 random integers between 1 and 50.
# Compute and print the minimum, maximum, and sum of the array.
array_min_max = np.random.randint(1, 50, 10)
print("Array:", array_min_max)
print("Minimum:", np.min(array_min_max))
print("Maximum:", np.max(array_min_max))
print("Sum:", np.sum(array_min_max))


# Exercise 16: Working with Dates and Times

# 1. Create an array of 10 dates starting from today with a daily frequency and print the array.
dates_daily = np.arange(np.datetime64('today'), np.datetime64('today') + np.timedelta64(10, 'D'))
print("Daily Dates:", dates_daily)

# 2. Create an array of 5 dates starting from January 1, 2022 with a monthly frequency and print the array.
dates_monthly = np.arange(np.datetime64('2022-01'), np.datetime64('2022-06'), np.timedelta64(1, 'M'))
print("Monthly Dates:", dates_monthly)

# 3. Create a 1D array with 10 random timestamps in the year 2023.
# Convert the timestamps to NumPy datetime64 objects and print the result.
random_timestamps = np.random.randint(0, 365, 10)  # Random days of the year
timestamps_2023 = np.datetime64('2023-01-01') + np.timedelta64(1, 'D') * random_timestamps
print("Random Timestamps in 2023:", timestamps_2023)


# Exercise 17: Creating Arrays with Custom Data Types

# 1. Create a 1D NumPy array of length 5 with a custom data type to store integers and their binary representation as strings.
dt = np.dtype([('number', np.int32), ('binary', 'S10')])
array_custom = np.array([(i, bin(i)) for i in range(5)], dtype=dt)
print("Custom Data Type Array:\n", array_custom)

# 2. Create a 2D NumPy array of shape (3, 3) with a custom data type to store complex numbers.
# Initialize the array with some complex numbers and print the array.
complex_dtype = np.dtype([('real', np.float64), ('imag', np.float64)])
array_complex = np.array([[(1, 2), (3, 4), (5, 6)],
                          [(7, 8), (9, 10), (11, 12)],
                          [(13, 14), (15, 16), (17, 18)]], dtype=complex_dtype)
print("Complex Number Array:\n", array_complex)

# 3. Create a structured array to store information about books with fields: title (string), author (string), and pages (integer).
# Add information for three books and print the structured array.
book_dtype = np.dtype([('title', 'S20'), ('author', 'S20'), ('pages', np.int32)])
books = np.array([('Dune', 'Frank Herbert', 412),
                  ('1984', 'George Orwell', 328),
                  ('Brave New World', 'Aldous Huxley', 288)], dtype=book_dtype)
print("Book Structured Array:\n", books)
