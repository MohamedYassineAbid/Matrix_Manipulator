import csv
import random

def generate_matrix(rows, cols):
    """
    Generate a matrix with random integer values.

    :param rows: Number of rows in the matrix.
    :param cols: Number of columns in the matrix.
    :return: A 2D list representing the matrix.
    """
    return [[random.randint(0, 100) for _ in range(cols)] for _ in range(rows)]

def save_matrix_to_csv(matrix, filename):
    """
    Save the matrix to a CSV file.

    :param matrix: A 2D list representing the matrix.
    :param filename: The name of the CSV file to save the matrix.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)

if __name__ == "__main__":
    # Define the size of the matrix
    rows = 500
    cols = 500

    # Generate the matrix
    matrix = generate_matrix(rows, cols)

    # Print the matrix (optional)
    for row in matrix:
        print(row)

    # Save the matrix to a CSV file
    filename = 'matrix.csv'
    save_matrix_to_csv(matrix, filename)
    print(f"Matrix saved to {filename}")