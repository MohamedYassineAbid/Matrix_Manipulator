from enum import Enum
import time
import streamlit as st
import algorithmes
import pandas as pd
import numpy as np

class MatrixType(Enum):
    SQUARE = "Square"
    SYMMETRIC = "Symmetric"
    DIAGONAL = "Diagonal"
    BAND = "Band"
    IDENTITY = "Identity"

class AlgorithmType(Enum):
    NONE = "None"
    TRANSPOSE = "Transpose"
    DETERMINANT = "Determinant"
    INVERSE = "Inverse"
    CHOLESKY = "Cholesky"
    POSITIVITY = "Positivity"

class InputType(Enum):
    MANUAL_INPUT = "Manual Input"
    RANDOM = "Random"
    CSV_UPLOAD = "CSV Upload"
    
    
    
# Convert a matrix to LaTeX format
def matrix_to_latex(matrix):
    latex_str = "\\begin{bmatrix}"
    for row in matrix:
        latex_str += " & ".join(map(str, row)) + " \\\\ "
    latex_str += "\\end{bmatrix}"
    return latex_str
# function to save the matrix to a CSV file

def save_matrix_to_csv(matrix):
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
        df = pd.DataFrame(matrix)
        csv_data = df.to_csv(index=False, header=False)
        return csv_data
    else:
        st.error("Matrix is not in a valid 2D format.")
        return None
# function to display the download button for the CSV file
def matrix_download_button(result):
    csv_file = save_matrix_to_csv(np.round(result, 2))
    if csv_file:
        st.download_button(
            label="Download Processed Matrix",
            data=csv_file,
            file_name="processed_matrix.csv",
            mime="text/csv"
        )

###### 
# Apply the chosen algorithm on the matrix
# remove [],[] from the return of the function if the function possed steps and descriptions
######
def apply_algorithm(matrix, algorithm):
    if algorithm == AlgorithmType.TRANSPOSE.value:
        return algorithmes.transposer(matrix), [], []
    elif algorithm == AlgorithmType.DETERMINANT.value:
        if algorithmes.isSquare(matrix):
            return algorithmes.Determinant(matrix), [], []
        else:
            return "Matrix must be square for Determinant!", [], []
    elif algorithm == AlgorithmType.INVERSE.value:
        if algorithmes.isSquare(matrix):
            return algorithmes.Inverse(matrix), [], []
        else:
            return "Matrix must be square for Inverse!", [], []
    elif algorithm == AlgorithmType.POSITIVITY.value:
        if algorithmes.isSquare(matrix) and algorithmes.isSymmetric(matrix) and algorithmes.is_positive_definite(matrix):
            return "Matrix is positive definite", [], []
        else:
            return "Matrix is not positive definite", [], []
    elif algorithm == AlgorithmType.CHOLESKY.value:
        if algorithmes.isSymmetric(matrix) and algorithmes.isSquare(matrix) and algorithmes.is_positive_definite(matrix):
            lower, steps, descriptions = algorithmes.cholesky(matrix)
            return lower, steps, descriptions
        else:
            return "Matrix must be square , symmetric  and positive definite for Cholesky Decomposition!", [], []
    else:
        return None, [], []

# Clear the matrix from session state
def clear_matrix():
    if 'matrix' in st.session_state:
        del st.session_state.matrix
        
        
# Main function to render the Streamlit app
def show_home():
    st.title("Matrix Operations with File Upload and Manual Input")
    st.sidebar.header("Matrix Settings")
    typeOfInput = [matrix_type.value for matrix_type in InputType]
    if 'matrix' not in st.session_state:
        st.session_state.matrix = None
    if 'input_type' not in st.session_state:
        st.session_state.input_type = None  
    if 'algorithm' not in st.session_state:
        st.session_state.algorithm = None  
    if 'generate_button' not in st.session_state:
        st.session_state.generate_button = False  

    if st.sidebar.button("Clear Matrix"):
        clear_matrix()
        st.session_state.matrix = None
        st.session_state.input_type = None
        st.session_state.algorithm = None
        st.session_state.generate_button = False  
        st.rerun()

    
    # Matrix input type selection
    input_type = st.sidebar.radio("Select Matrix Input Type", typeOfInput)
    if (st.session_state.input_type == InputType.CSV_UPLOAD.value and input_type != InputType.CSV_UPLOAD.value) or (st.session_state.input_type == InputType.RANDOM.value and input_type != InputType.RANDOM.value) or (st.session_state.input_type == InputType.MANUAL_INPUT.value and input_type != InputType.MANUAL_INPUT.value):
        st.session_state.matrix = None  

    st.session_state.input_type = input_type
    if input_type == InputType.RANDOM.value:
        
        typeOfMatrix = [matrix_type.value for matrix_type in MatrixType]
        element_type = st.sidebar.radio("Select Element Type", ["int", "float"])
        matrix_type = st.sidebar.selectbox("Select Matrix Type", typeOfMatrix)
        if matrix_type in typeOfMatrix:
            rows = cols = st.sidebar.number_input("Matrix Size (n x n)", min_value=1, max_value=10, value=3)
            if matrix_type == MatrixType.BAND.value:
                lower_bandwidth = st.sidebar.number_input("Lower Bandwidth", min_value=0, max_value=rows-1, value=1)
                upper_bandwidth = st.sidebar.number_input("Upper Bandwidth", min_value=0, max_value=rows-1, value=1)
            if matrix_type == MatrixType.SYMMETRIC.value:
                yesorno=st.sidebar.radio("Do you want the matrix to be positive definite (No ==> mean random can be positive and it cant be)",["Yes","No"])
            col1, col2 = st.sidebar.columns(2)  
            if matrix_type != MatrixType.IDENTITY.value:    
                min_val = col1.number_input("Min Value", value=0)
                max_val = col2.number_input("Max Value", value=10)
            generate_button = st.sidebar.button("Generate Matrix")

        if generate_button:
            if matrix_type == MatrixType.SQUARE.value:
                st.session_state.matrix = algorithmes.generate_square_matrix(rows, element_type, min_val, max_val)
            elif matrix_type == MatrixType.SYMMETRIC.value and yesorno=="No":
                st.session_state.matrix = algorithmes.generate_symmetric_matrix(rows, element_type, min_val, max_val)
            elif matrix_type == MatrixType.SYMMETRIC.value and yesorno=="Yes":
                st.session_state.matrix = algorithmes.generate_positive_definite_matrix(rows, element_type, min_val, max_val)
            elif matrix_type == MatrixType.DIAGONAL.value:
                st.session_state.matrix = algorithmes.generate_diagonal_matrix(rows, element_type, min_val, max_val)
            elif matrix_type == MatrixType.BAND.value:
                st.session_state.matrix = algorithmes.generate_band_matrix(rows, element_type, lower_bandwidth, upper_bandwidth, min_val, max_val)
            elif matrix_type == MatrixType.IDENTITY.value:
                st.session_state.matrix = algorithmes.generate_identity_matrix(rows)
        if st.session_state.matrix is None:
            st.write("### Waiting for matrix generation...")
                


    elif input_type == InputType.CSV_UPLOAD.value:
        st.write("### Upload a Matrix File (CSV):")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state.matrix = df.to_numpy()
                st.write("### Matrix from CSV File:")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading file: {e}")


    elif input_type == InputType.MANUAL_INPUT.value:
        rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=10, value=3)
        cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=10, value=3)
        st.write("### Enter Matrix Values Manually")
        st.session_state.matrix = np.zeros((rows, cols))
        for i in range(rows):
            cols_input = st.columns(cols)
            for j in range(cols):
                cell_key = f"manual_cell_{i}_{j}"
                if cell_key not in st.session_state:
                    st.session_state[cell_key] = "0.0"  
                st.session_state.matrix[i][j] = float(cols_input[j].text_input(f"## M[{i + 1}][{j + 1}]",value=st.session_state[cell_key],key=cell_key))

    # Matrix operation selection, always visible
    st.sidebar.header("Matrix Operations")
    algorithm_type = [algorithm.value for algorithm in AlgorithmType]
    algorithm = st.sidebar.selectbox("Choose an Algorithm", algorithm_type)
    
    
    # Apply selected algorithm and display results for csv
    if st.session_state.matrix is not None and input_type == InputType.CSV_UPLOAD.value:
        if algorithm != AlgorithmType.NONE.value:
            result, _, _ = apply_algorithm(st.session_state.matrix, algorithm)
            
            if isinstance(result, np.ndarray):
                csv_file = save_matrix_to_csv(result)
                
                if csv_file:
                    st.write(f"### Result of {algorithm}:")
                    st.dataframe(pd.DataFrame(result))
                    matrix_download_button(result)
            else:
                st.write(f"#### {result}") 

    
    
    # Display matrix in LaTeX format if available for non csv

    if st.session_state.matrix is not None and input_type != InputType.CSV_UPLOAD.value:
        st.write("### Matrix in LaTeX Format:")
        st.latex(matrix_to_latex(st.session_state.matrix))

        # Apply selected algorithm and display results
        if algorithm != AlgorithmType.NONE.value:
            result, steps, descriptions = apply_algorithm(st.session_state.matrix, algorithm)

            if result is not None and (steps is  None  or len(steps)==0 ) and (descriptions is  None or len(descriptions)==0):
                st.write(f"### Result of {algorithm}:")
                if isinstance(result, np.ndarray):
                    st.latex(matrix_to_latex(result)) 
                    matrix_download_button(result)
                else:
                    st.write(f"#### {result}") 

            else :
                
                if  isinstance(result, np.ndarray):
                    st.write(f"### Steps of {algorithm}:")
                    for step, description in zip(steps, descriptions):
                        st.write(description)
                        st.latex(matrix_to_latex(step))
                    st.write(f"### Result of {algorithm}:")
                    st.latex(matrix_to_latex(result)) 
                    matrix_download_button(result) 
