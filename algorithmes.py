from math import *
import random
import numpy as np

# Cholesky decomposition algorithm


def cholesky(A):
    """Perform Cholesky decomposition of a symmetric positive-definite matrix A."""
    A = np.copy(A)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_k = np.dot(L[i, :j], L[j, :j])
            if i == j:  # Diagonal elements
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]

    return L,[],[]

def forward_substitution(L, b):
    """Solve Ly = b for y using forward substitution."""
    L = np.copy(L)
    b = np.copy(b)
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        sum_k = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - sum_k) / L[i, i]

    return y

def back_substitution(U, y):
    """Solve Ux = y for x using backward substitution."""
    U = np.copy(U)
    y = np.copy(y)
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_k = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - sum_k) / U[i, i]

    return x

def resolution(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    A = np.copy(A)
    b = np.copy(b)

    L = cholesky(A)
    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    return x


type_mapping = {"int": int, "float": float}

# Generate a symmetric matrix with specified value range

def generate_symmetric_matrix(n, element_type):
    # Initialize a square matrix of size n x n
    matrix = np.zeros((n, n))
    min_val=-50
    max_val=50
    for i in range(n):
        for j in range(i, n):
            random_value = random.randint(min_val, max_val)
            matrix[i][j] = random_value
            matrix[j][i] = random_value
    return np.round(matrix.astype(type_mapping[element_type]), 2)


# Genetate a positive definite matrix
def generate_positive_definite_matrix(n, element_type):
    min_val=0
    max_val=50
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
    return np.round(np.random.uniform(min_val, max_val, (n, n)), 2).astype(
        type_mapping[element_type]
    )


# Generate an identity matrix
def generate_identity_matrix(n):
    return np.identity(n).astype(int)


# Generate a band matrix with specified bandwidths and value range
def generate_band_matrix(n, element_type, lower_bandwidth, upper_bandwidth, low=0, high=10):
    A = np.zeros((n, n))
    # Fill the matrix with random values within the specified bandwidths
    for i in range(n):
        for j in range(max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)):
            A[i, j] = float(np.round(np.random.uniform(low, high), 2))

    return A.astype(type_mapping[element_type])


################################################################################
################################################################################
################################################################################


# Matrix Minor
def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]



# check if defined positive
def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)
import numpy as np

# Gaussian elimination method with error message instead of raising errors
def gauss_elimination(A):
    A = A.copy()
    n = A.shape[0]
    for k in range(n-1):
        max_row = k
        max_value = abs(A[k, k])
        for i in range(k+1, n):
            if abs(A[i, k]) > max_value:
                max_value = abs(A[i, k])
                max_row = i
        
        # If pivot element is zero, matrix is singular, return message
        if max_value == 0:
            return A, "Matrix is singular or nearly singular.(det=0)"
        
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
        
        for i in range(k+1, n):
            if A[i, k] != 0.0:
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
    
    return A, None  # Return None if no error







#determinant

def Determinant(a):
    a,b=gauss_elimination(a) 
    if b :
        return b           
    return np.prod(np.diag(a))


def Inverse(A):
    n = A.shape[0]
    augmented_matrix = np.hstack((A, np.eye(n)))
    
    U, error_msg = gauss_elimination(augmented_matrix)
    
    if error_msg:
        return error_msg  
    
    inverse_matrix = U[:, n:]
    
    return inverse_matrix

def transposer(matrix):
    return np.array([[row[i] for row in matrix] for i in range(len(matrix[0]))])


def isSymmetric(matrix):
    try:
        return np.all(matrix == transposer(matrix))
    except:
        return False


# check if the matrix is square
def isSquare(matrix):
    return matrix.shape[0] == matrix.shape[1]

def is_diagonal(matrix):
    n = len(matrix) 
    
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] != 0:
                return False  
    
    return True
def is_identity(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if i == j and matrix[i][j] != 1:
                return False
            if i != j and matrix[i][j] != 0:
                return False
    return True  

def mat_profile(matrix):
    profile = []
    if isSquare(matrix):
        profile.append("Square")
        if isSymmetric(matrix):
            profile.append("Symmetric")
        else:
            profile.append("Not Symmetric")
        if is_identity(matrix) :
            profile.append("Identity")
        if is_diagonal(matrix) :
            profile.append("Diagonal")
        else:
            profile.append("Not Diagonal")
        
    else :
        profile.append("Not Square")
    return profile