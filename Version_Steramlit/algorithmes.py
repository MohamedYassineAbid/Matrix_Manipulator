import math
import numpy as np

def Cholesky_Decomposition(matrix):
    n = matrix.shape[0]
    lower = np.zeros((n, n))
    steps = []
    descriptions = []

    for i in range(n):
        for j in range(i + 1):
            sum1 = 0
            if j == i:
                for k in range(j):
                    sum1 += lower[j][k] ** 2
                try:    
                    lower[j][j] = math.sqrt(matrix[j][j] - sum1)
                    steps.append(lower.copy())
                    descriptions.append(f"Step {i + 1}: Calculate L[{j}][{j}] = sqrt(A[{j}][{j}] - sum) = sqrt({matrix[j][j]} - {sum1}) = {lower[j][j]}")
                except :
                    print("er")
            else:
                for k in range(j):
                    sum1 += lower[i][k] * lower[j][k]
                lower[i][j] = (matrix[i][j] - sum1) / lower[j][j]
                steps.append(lower.copy())
                descriptions.append(f"Step {i + 1}.{j + 1}: Calculate L[{i}][{j}] = (A[{i}][{j}] - sum) / L[{j}][{j}] = ({matrix[i][j]} - {sum1}) / {lower[j][j]} = {lower[i][j]}")

    return lower, steps, descriptions

def Transpose(matrix):
    return matrix.T, [], []

def Determinant(matrix):
    return np.linalg.det(matrix), [], []

def Inverse(matrix):
    return (np.linalg.inv(matrix), [], []) if np.linalg.det(matrix) != 0 else ("Matrix is singular!", [], [])

def transposer(matrix):
    return matrix.T, [], []


