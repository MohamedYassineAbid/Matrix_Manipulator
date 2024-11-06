import streamlit as st
import algorithmes
import pandas as pd
import numpy as np

# Convert a matrix to LaTeX format
def matrix_to_latex(matrix):
    latex_str = "\\begin{bmatrix}"
    for row in matrix:
        latex_str += " & ".join(map(str, row)) + " \\\\ "
    latex_str += "\\end{bmatrix}"
    return latex_str

# Apply the chosen algorithm on the matrix
def apply_algorithm(matrix, algorithm):
    if algorithm == "Transpose":
        return algorithmes.Transpose(matrix), [], []
    elif algorithm == "Determinant":
        if algorithmes.isSquare(matrix):
            return algorithmes.Determinant(matrix), [], []
        else:
            return "Matrix must be square for Determinant!", [], []
    elif algorithm == "Inverse":
        if algorithmes.isSquare(matrix):
            return algorithmes.Inverse(matrix), [], []
        else:
            return "Matrix must be square for Inverse!", [], []
    elif algorithm == "Positivity":
        if algorithmes.isSquare(matrix) and algorithmes.isSymmetric(matrix) and algorithmes.is_positive_definite(matrix):
            return "Matrix is positive definite", [], []
        else:
            return "Matrix is not positive definite", [], []
    elif algorithm == "Cholesky":
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
# Main function to render the Streamlit app
def show_home():
    st.title("Matrix Operations with File Upload and Manual Input")
    st.sidebar.header("Matrix Settings")

    # Initialize session state for the matrix if it doesn't exist
    if 'matrix' not in st.session_state:
        st.session_state.matrix = None

    # Add the Clear button to reset the matrix
    if st.sidebar.button("Clear Matrix"):
        clear_matrix()
        st.experimental_rerun()  # Refresh the app to reflect changes

    # Matrix input type selection
    input_type = st.sidebar.radio("Select Matrix Input Type", ["Random", "CSV Upload", "Manual Input"])

    # Matrix type selection and generation if "Random" is selected
    if input_type == "Random":
        typeOfMatrix = ["Square", "Symmetric", "Diagonal", "Band", "Identity"]
        element_type = st.sidebar.radio("Select Element Type", ["int", "float"])
        matrix_type = st.sidebar.selectbox("Select Matrix Type", typeOfMatrix)
        if matrix_type in typeOfMatrix:
            rows = cols = st.sidebar.number_input("Matrix Size (n x n)", min_value=1, max_value=10, value=3)
            if matrix_type == "Band":
                lower_bandwidth = st.sidebar.number_input("Lower Bandwidth", min_value=0, max_value=10, value=1)
                upper_bandwidth = st.sidebar.number_input("Upper Bandwidth", min_value=0, max_value=10, value=1)
            if matrix_type == "Symmetric":
                yesorno=st.sidebar.radio("Do you want the matrix to be positive definite (No ==> mean random can be positive and it cant be)",["Yes","No"])
            col1, col2 = st.sidebar.columns(2)
            min_val = col1.number_input("Min Value", value=0)
            max_val = col2.number_input("Max Value", value=10)
            generate_button = st.sidebar.button("Generate Matrix")

            # Generate random matrix based on type
            if generate_button:
                if matrix_type == "Square":
                    st.session_state.matrix = algorithmes.generate_square_matrix(rows, element_type, min_val, max_val)
                elif matrix_type == "Symmetric" and yesorno=="No":
                    st.session_state.matrix = algorithmes.generate_symmetric_matrix(rows, element_type, min_val, max_val)
                elif matrix_type == "Symmetric" and yesorno=="Yes":
                    st.session_state.matrix = algorithmes.generate_positive_definite_matrix(rows, element_type, min_val, max_val)
                elif matrix_type == "Diagonal":
                    st.session_state.matrix = algorithmes.generate_diagonal_matrix(rows, element_type, min_val, max_val)
                elif matrix_type == "Band":
                    st.session_state.matrix = algorithmes.generate_band_matrix(rows, element_type, lower_bandwidth, upper_bandwidth, min_val, max_val)
                elif matrix_type == "Identity":
                    st.session_state.matrix = algorithmes.generate_identity_matrix(rows)

    elif input_type == "CSV Upload":
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

    elif input_type == "Manual Input":
        rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=10, value=3)
        cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=10, value=3)
        st.write("### Enter Matrix Values Manually")
        st.session_state.matrix = np.zeros((rows, cols))
        for i in range(rows):
            cols_input = st.columns(cols)
            for j in range(cols):
                st.session_state.matrix[i][j] = float(cols_input[j].text_input(f"Row {i + 1}, Col {j + 1}", value="0"))

    # Matrix operation selection, always visible
    st.sidebar.header("Matrix Operations")
    algorithm_type = ["Transpose", "Determinant", "Inverse", "Cholesky", "Positivity"]
    algorithm = st.sidebar.selectbox("Choose an Algorithm", algorithm_type)

    # Display matrix in LaTeX format if available
    if st.session_state.matrix is not None:
        st.write("### Matrix in LaTeX Format:")
        st.latex(matrix_to_latex(st.session_state.matrix))

        # Apply selected algorithm and display results
        if algorithm != "None":
            st.write(f"### Result of {algorithm}:")
            result, steps, descriptions = apply_algorithm(st.session_state.matrix, algorithm)
            if result is not None:
                if isinstance(result, np.ndarray):
                    st.latex(matrix_to_latex(result))  # Display resulting matrix in LaTeX format
                else:
                    st.write(f"#### {result}")  # Display error or result in text form if not an array

                # Display additional information for Cholesky
                if algorithm == "Cholesky" and isinstance(result, np.ndarray):
                    st.write("### Steps of Cholesky Decomposition:")
                    for step, description in zip(steps, descriptions):
                        st.write(description)
                        st.latex(matrix_to_latex(step))  # Display each step in LaTeX format