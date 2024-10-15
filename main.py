import numpy as np
import streamlit as st
import pandas as pd
import algorithmes



def matrix_to_latex(matrix):
    latex_str = "\\begin{bmatrix}"
    for row in matrix:
        latex_str += " & ".join(map(str, row)) + " \\\\ "
    latex_str += "\\end{bmatrix}"
    return latex_str

def apply_algorithm(matrix, algorithm):
    if algorithm == "Transpose":
        return algorithmes.Transpose(matrix)
    elif algorithm == "Determinant":
        return algorithmes.Determinant(matrix)
    elif algorithm == "Inverse":
        return algorithmes.Inverse(matrix)
    elif algorithm == "cholesky":
        if matrix.shape[0] == matrix.shape[1]:
            if np.allclose(matrix, matrix.T):
                lower, steps, descriptions = algorithmes.Cholesky_Decomposition(matrix)
                return lower, steps, descriptions
            else:
                return "Matrix is not symmetric!", [], []
        else:
            return "Matrix must be square!", [], []
    else:
        return None, [], []


st.title("Matrix Input with File Upload and Algorithms")
st.sidebar.header("Matrix Settings")
rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=10, value=3)
cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=10, value=3)

st.sidebar.header("Matrix Operations")
algorithm = st.sidebar.selectbox("Choose an algorithm", ["None", "cholesky", "Transpose", "Determinant", "Inverse"])

st.write("### Upload a Matrix File (CSV) or Enter Manually:")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        matrix = df.to_numpy()
        st.write("### Matrix from File:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.write("### Enter Matrix Values Manually:")
    matrix = np.zeros((rows, cols))

    for i in range(rows):
        cols_input = st.columns(cols)
        for j in range(cols):
            matrix[i][j] = float(cols_input[j].text_input(f"Row {i + 1}, Col {j + 1}", value="0"))

st.write("### Matrix in LaTeX Format:")
st.latex(matrix_to_latex(matrix))

if algorithm != "None":
    st.write(f"### Result of {algorithm}:")
    result, steps, descriptions = apply_algorithm(matrix, algorithm)

    if isinstance(result, np.ndarray):
        st.write(result)
        st.latex(matrix_to_latex(result))

        if algorithm == "cholesky":
            st.write("### Steps of Cholesky Decomposition:")
            for step, description in zip(steps, descriptions):
                st.write(description)
                st.latex(matrix_to_latex(step)) 
    else:
        st.write(result)
