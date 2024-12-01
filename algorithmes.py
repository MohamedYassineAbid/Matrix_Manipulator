from math import *
import random
import numpy as np

# Cholesky decomposition algorithm


def cholesky(A):
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower triangular matrix, L, along with the
    steps and descriptions of the calculations."""
    n = len(A)
    L = np.zeros((n, n))
    steps = []  
    descriptions = []  

    # Perform the Cholesky decomposition
    for j in range(n):
        L[j][j] = A[j][j]
        
        for k in range(j):
            L[j][j] -= L[j][k] ** 2
        
        try:
            L[j][j] = sqrt(L[j][j])
        except ValueError as e:
            return "Matrix is not positive definite.",[],[]

        for i in range(j + 1, n):
            L[i][j] = A[i][j]
            for k in range(j):
                L[i][j] -= L[i][k] * L[j][k]
            try:
                if L[j][j] == 0:
                    return f"Division by zero encountered at L[{i+1}][{j+1}].",[],[]
                L[i][j] /= L[j][j]
            except ZeroDivisionError as e:
                return str(e), [], []
            
        steps.append(np.round(L, 2))
        descriptions.append(
            f"Step {j+1}: L_{j+1}{j+1} = {L[j][j]}\n"
            + "".join(
                [
                    f"L_{i+1}{j+1} = {L[i][j]}\n"
                    for i in range(j + 1, n)
                ]
            )
        )
        
            

    return np.round(L, 2), np.round(steps, 2), descriptions
# Resolution method for solving system of equations with message output instead of errors
def resolution(L,b):
    L,_,_=cholesky(L)
    b=b.copy()
    L=np.array(L, float)
    U=transposer(L)
    b=np.array(b, float)

    n,_=np.shape(L)
    y=np.zeros(n)
    x=np.zeros(n)

    # Forward substitution
    for i in range(n):
        sumj=0
        for j in range(i):
            sumj += L[i,j]*y[j]
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Backward substitution  
    for i in range(n-1, -1, -1):
        sumj=0
        for j in range(i+1,n):
            sumj += U[i,j] * x[j]
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return np.round(x,4)
        


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
def leading_principal_minors(matrix):
    n = len(matrix)
    minors = []
    for k in range(1, n + 1):
        submatrix = [row[:k] for row in matrix[:k]]  
        minors.append(Determinant(np.array(submatrix)))
    return minors


def is_positive_definite(matrix):
    
    if not isSymmetric(matrix):
        return False, "Matrix is not symmetric"
    
    minors = leading_principal_minors(matrix)
    for minor in minors:
        if minor <= 0:
            return False, "Matrix is not positive definite"
    
    return True, None

# Gaussian elimination method with error message instead of raising errors
def gauss_elimination(A):
    # Ensure the matrix is of float type to avoid integer division issues
    A = A.astype(float)
    n = A.shape[0]

    # Forward elimination with partial pivoting
    for k in range(n-1):
        # Find the row with the largest pivot element
        max_row = k
        max_value = abs(A[k, k])
        for i in range(k+1, n):
            if abs(A[i, k]) > max_value:
                max_value = abs(A[i, k])
                max_row = i
        
        # If pivot element is zero, matrix is singular
        if max_value == 0:
            return None, "Matrix is singular or nearly singular (det=0)"
        
        # Swap the rows
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
        
        # Eliminate below pivot
        for i in range(k+1, n):
            if A[i, k] != 0.0:
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]

    return A,None






#determinant

def Determinant(a):
    a,b=gauss_elimination(a) 
    if b :
        return b           
    return np.prod(np.diag(a))


def Inverse(A):
    n = A.shape[0]
    augmented_matrix = np.hstack((A.astype(float), np.eye(n)))
    for k in range(n):
        max_row = np.argmax(np.abs(augmented_matrix[k:n, k])) + k
        if augmented_matrix[max_row, k] == 0:
            return None, "Matrix is singular or nearly singular (det=0)"
        
        augmented_matrix[[k, max_row]] = augmented_matrix[[max_row, k]]
        
        augmented_matrix[k] /= augmented_matrix[k, k]
        
        for i in range(k+1, n):
            factor = augmented_matrix[i, k]
            augmented_matrix[i] -= factor * augmented_matrix[k]
    
    for k in range(n-1, -1, -1):
        for i in range(k-1, -1, -1):
            factor = augmented_matrix[i, k]
            augmented_matrix[i] -= factor * augmented_matrix[k]
    
    inverse_matrix = augmented_matrix[:, n:]
    
    return np.round(inverse_matrix, 5)
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