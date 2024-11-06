from math import *
import numpy as np

# Cholesky decomposition algorithm

def cholesky(A):
    """Performs a Cholesky decomposition of A, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower triangular matrix, L, along with the 
    steps and descriptions of the calculations."""
    n = len(A)

    # Create zero matrix for L
    L = np.zeros((n, n))

    steps = []  # To hold intermediate matrices
    descriptions = []  # To hold descriptions of each step

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))

            if i == k:  # Diagonal elements
                L[i][k] = sqrt(A[i][i] - tmp_sum)
                # Add the current state of L and the description
                steps.append(L.copy())
                descriptions.append(f"##### Calculated L[{i},{k}] = √(A[{i},{i}] - Σ(L[{i},j]²) for j=0 to {k-1}.")
            else:
                L[i][k] = (1.0 / L[k][k]) * (A[i][k] - tmp_sum)
                # Add the current state of L and the description
                steps.append(L.copy())
                descriptions.append(f" ##### Calculated L[{i},{k}] = (A[{i},{k}] - Σ(L[{i},j] * L[{k},j]) for j=0 to {k-1}) / L[{k},{k}].")

    return np.round(L, 2), np.round(steps,2), descriptions





type_mapping = {
    "int": int,
    "float": float
}

# Generate a symmetric matrix with specified value range
def generate_symmetric_matrix(n, element_type, min_val, max_val):
    random_matrix = np.random.uniform(min_val, max_val, (n, n))
    symmetric_matrix = random_matrix + Transpose(random_matrix)
    return np.round(symmetric_matrix, 2).astype(type_mapping[element_type])

#Genetate a positive definite matrix
def generate_positive_definite_matrix(n, element_type, min_val=0, max_val=10):
    random_matrix = np.random.uniform(min_val, max_val, (n, n))
    
    positive_definite_matrix = np.dot(random_matrix, random_matrix.T)
    
    positive_definite_matrix = np.round(positive_definite_matrix, 2)
    return positive_definite_matrix.astype(type_mapping[element_type])


# Generate a diagonal matrix with specified value range
def generate_diagonal_matrix(n, element_type, min_val, max_val):
    diagonal_values = np.random.uniform(min_val, max_val, n)
    return np.diag(np.round(diagonal_values, 2).astype(type_mapping[element_type]))

# Generate a square matrix with specified value range
def generate_square_matrix(n, element_type, min_val, max_val):
    return np.round(np.random.uniform(min_val, max_val, (n, n)), 2).astype(type_mapping[element_type])

# Generate an identity matrix
def generate_identity_matrix(n):
    return np.identity(n).astype(int)

# Generate a band matrix with specified bandwidths and value range
def generate_band_matrix(n,element_type, lower_bandwidth, upper_bandwidth, low=0, high=10):
    A = np.zeros((n, n))
    # Fill the matrix with random values within the specified bandwidths
    for i in range(n):
        for j in range(max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)):
            A[i, j] = float(np.round(np.random.uniform(low, high), 2))
            
    return A
# Matrix Minor
def getMatrixMinor(m, p):
     if (p>1) : return [row[:p] + row[p + 1:] for row in (m[:p] + m[p + 1:])]
#check if defined positive
def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

    
def Transpose(matrix):
    return matrix.T

def Determinant(matrix):
    return np.round(np.linalg.det(matrix),2)

def Inverse(matrix):
    return ((np.round(np.linalg.inv(matrix),2))) if np.linalg.det(matrix) != 0 else ("Matrix is singular!")

#Produit de deux matrices
def produit(A, B):
    return [[sum(L[k] * B[k][j] for k in range(len(L))) for j in range(len(B[0]))] for L in A]

#Addition de deux matrices
def addition(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

#CHECK IF THE MATRIX IS SYMMETRIC
def isSymmetric(matrix):
    return (matrix == matrix.T).all()

#check if the matrix is square
def isSquare(matrix):
    return matrix.shape[0] == matrix.shape[1]