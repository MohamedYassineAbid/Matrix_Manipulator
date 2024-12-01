from enum import Enum
import streamlit as st
import algorithmes
import pandas as pd
import numpy as np
from streamlit_chat import message
import creds as creds
import io
import google.generativeai as genai
from datetime import datetime
import json

class MatrixType(Enum):
    SQUARE = "Square"
    SYMMETRIC = "Symmetric"
    DIAGONAL = "Diagonal"
    BAND = "Band"
    IDENTITY = "Identity"

class Dimension():
    rows=0
    cols=0

class AlgorithmType(Enum):
    NONE = "None"
    profile = "profile"
    
    TRANSPOSE = "Transpose"
    DETERMINANT = "Determinant"
    INVERSE = "Inverse"
    CHOLESKY = "Cholesky"
    POSITIVITY = "Positivity"
    CHOLESKY_RESOLUTION="Resolution" 


class InputType(Enum):
    MANUAL_INPUT = "Manual Input"
    RANDOM = "Random"
    CSV_UPLOAD = "CSV Upload"
    

def matrix_to_latex(matrix):
    latex_str = "\\begin{bmatrix}\n"
    
    for row in matrix:
        row = row if isinstance(row, (list, np.ndarray)) else [row]
        
        latex_str += " & ".join(map(str, row)) + " \\\\ \n"
    
    latex_str += "\\end{bmatrix}"
    return latex_str


def save_matrix_to_csv(matrix):
    df = pd.DataFrame(np.round(matrix, 2))
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False,header=False)
    buffer.seek(0) 
    return buffer.getvalue() 

# Function to display the download button
def download_csv(csv_content):
    csv_bytes = csv_content.encode() 
    
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="matrix.csv",
        mime="text/csv",
    )



# Apply the chosen algorithm on the matrix
def apply_algorithm(matrix, algorithm):
    
    if algorithm == AlgorithmType.CHOLESKY_RESOLUTION.value:
        b_matrix = handle_resolution(matrix)
        try:
            if algorithmes.isSquare(matrix):
                solution = algorithmes.resolution(matrix, b_matrix)
                return solution, [], []
            else:
                return "Matrix A must be square to solve AX = B!", [], []
        except np.linalg.LinAlgError as e:
            return f"Error in solving AX = B: {e}", [], []
    elif algorithm == AlgorithmType.profile.value:
        return algorithmes.mat_profile(matrix), [], []
    elif algorithm == AlgorithmType.TRANSPOSE.value:
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
        if (
            algorithmes.isSquare(matrix)
            and algorithmes.isSymmetric(matrix)
            and algorithmes.is_positive_definite(matrix)
        ):
            return "Matrix is positive definite", [], []
        else:
            return "Matrix is not positive definite", [], []
    elif algorithm == AlgorithmType.CHOLESKY.value:
        if (
            algorithmes.isSymmetric(matrix)
            and algorithmes.isSquare(matrix)
            and algorithmes.is_positive_definite(matrix)
        ):
            lower, steps, descriptions = algorithmes.cholesky(matrix)
            return lower, steps, descriptions
        else:
            return (
                "Matrix must be square , symmetric  and positive definite for Cholesky Decomposition!",
                [],
                [],
            )
    else:
        return None, [], []


# Clear the matrix from session state
def clear_matrix():
    st.session_state.matrix = None
    st.session_state.input_type = None
    st.session_state.algorithm = None
    st.session_state.generate_button = False
    if "matrix" in st.session_state:
        del st.session_state.matrix
    st.rerun()



                
# Main function to render the Streamlit app
def generate_random_matrix(matrix_type, rows, element_type, min_val, max_val, lower_bandwidth=None, upper_bandwidth=None, yesorno=None):
    if matrix_type == MatrixType.SQUARE.value:
        return algorithmes.generate_square_matrix(rows, element_type, min_val, max_val)
    elif matrix_type == MatrixType.SYMMETRIC.value:
        if yesorno == "No":
            return algorithmes.generate_symmetric_matrix(rows, element_type)
        else:
            return algorithmes.generate_positive_definite_matrix(rows, element_type)
    elif matrix_type == MatrixType.DIAGONAL.value:
        return algorithmes.generate_diagonal_matrix(rows, element_type, min_val, max_val)
    elif matrix_type == MatrixType.BAND.value:
        return algorithmes.generate_band_matrix(rows, element_type, lower_bandwidth, upper_bandwidth, min_val, max_val)
    elif matrix_type == MatrixType.IDENTITY.value:
        return algorithmes.generate_identity_matrix(rows)
    return None

def handle_resolution(matrix):
    st.write("### Provide Matrix B for AX = B:")
    rows, cols = matrix.shape
    Dimension.rows=rows
    Dimension.cols=cols
    if (st.session_state.input_type == InputType.CSV_UPLOAD.value):
        return handle_csv_b_input(rows)
    else:
        st.latex(matrix_to_latex(matrix))
        return handle_manual_b_input(rows)
    
def handle_csv_b_input(rows):
    st.write("### Upload a Matrix B File (CSV):")
    uploaded_file = st.file_uploader("Upload another CSV file", type=["csv"], key="csv_file_uploader_2")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[0] != rows:
                st.error("Number of rows in B matrix must match number of rows in A matrix!")
                return None
            return df.to_numpy()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None
def handle_manual_b_input(rows):
    b_matrix = np.zeros((rows, 1))
    b_cols_input = st.columns(1)
    for i in range(rows):
        b_matrix[i][0] = b_cols_input[0].number_input(f"B[{i + 1}]", value=0.0,step=1.0)
    return b_matrix

def handle_matrix_input(input_type):
    if input_type == InputType.RANDOM.value:
        return handle_random_matrix_input()
    elif input_type == InputType.CSV_UPLOAD.value:
        return handle_csv_upload()
    elif input_type == InputType.MANUAL_INPUT.value:
        return handle_manual_input()
    
def handle_random_matrix_input():
    typeOfMatrix = [matrix_type.value for matrix_type in MatrixType]
    element_type = st.sidebar.radio("Select Element Type", ["int", "float"])
    matrix_type = st.sidebar.selectbox("Select Matrix Type", typeOfMatrix)
    rows = cols = st.sidebar.number_input("Matrix Size (n x n)", min_value=1, max_value=10, value=3)
    Dimension.rows=rows
    Dimension.cols=cols

    lower_bandwidth, upper_bandwidth = None, None
    yesorno = None
    col1, col2 = st.sidebar.columns(2)
    if matrix_type == MatrixType.BAND.value:
        lower_bandwidth = col1.number_input("Lower Bandwidth", min_value=0, max_value=rows - 1, value=1)
        upper_bandwidth = col2.number_input("Upper Bandwidth", min_value=0, max_value=rows - 1, value=1)

    if matrix_type == MatrixType.SYMMETRIC.value:
        yesorno = st.sidebar.radio(
            "Do you want the matrix to be positive definite?", ["Yes", "No"]
        )
    if matrix_type == MatrixType.IDENTITY.value or matrix_type == MatrixType.SYMMETRIC.value:
        min_val = max_val = None        
        
    else :   
        min_val = col1.number_input("Min Value", value=0)
    
        max_val = col2.number_input("Max Value", value=10)
    generate_button = st.sidebar.button("Generate Matrix")

    if generate_button:
        matrix = generate_random_matrix(matrix_type, rows, element_type, min_val, max_val, lower_bandwidth, upper_bandwidth, yesorno)
        st.session_state.matrix = matrix


def handle_csv_upload():
    st.write("### Upload a Matrix File (CSV):")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_file_uploader_1")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.session_state.matrix = df.to_numpy()
            st.write("### Matrix from CSV File:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")




def handle_manual_input():
    # Select matrix type
    matrix_type = st.sidebar.selectbox(
        "Select Matrix Type",
        ["Custom", "Symmetric", "Diagonal", "Identity"]
    )

    # Set matrix dimensions
    
    
    # Ensure square matrix for symmetric/diagonal/identity types
    if matrix_type in ["Symmetric", "Diagonal", "Identity"]:
        rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=5, value=3)
        cols = rows
        st.sidebar.info("Matrix forced to square for selected type.")
    else :
        rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=5, value=3)
        cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=5, value=3)

    Dimension.rows=rows
    Dimension.cols=cols
    st.write("### Enter Matrix Values")

    # Initialize matrix
    matrix = np.zeros((rows, cols))

    if matrix_type == "Custom":
        for i in range(rows):
            cols_input = st.columns(cols)
            for j in range(cols):
                matrix[i][j] = cols_input[j].number_input(
                    f"Row {i + 1}, Col {j + 1}", value=0.0, step=1.0, key=f"custom_{i}_{j}"
                )

    elif matrix_type == "Symmetric":
        for i in range(rows):
            cols_input = st.columns(cols)
            for j in range(i, cols):  
                matrix[i][j] = cols_input[j].number_input(
                    f"Row {i + 1}, Col {j + 1}", value=0.0, step=1.0, key=f"sym_{i}_{j}"
                )
                matrix[j][i] = matrix[i][j]  

    elif matrix_type == "Diagonal":
        # Input for diagonal elements only
        for i in range(rows):
            cols_input = st.columns(1)  
            matrix[i][i] = cols_input[0].number_input(
                f"Diagonal Element {i + 1}", value=0.0, step=1.0, key=f"diag_{i}"
            )

    elif matrix_type == "Identity":
        # Auto-fill identity matrix
        matrix = np.eye(rows)
        st.info("Identity Matrix auto-filled!")

    # Store and display matrix
    st.session_state.matrix = matrix


def save_the_matrix(result:str,algorithm_name:str)->None:
    #download the matrix in matrices folder name:user_type_time
    #open json file with the name of the username and append the matrix
    date=datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    date_replace=date.replace(':','_')
    path=f"assets/token/matrices/{st.session_state['username']}_{algorithm_name}_{date_replace}.csv"
    
    new_matrix={
        "Type": algorithm_name,
        "Height": str(Dimension.rows),
        "Width": str(Dimension.cols),
        "Date": date,
        "File": f"{st.session_state['username']}_{algorithm_name}_{date_replace}.csv"
                }
    np.savetxt(path,result,delimiter=',',fmt="%4f")
    
    with open(f"assets/token/matrices/{st.session_state['username']}.json","r") as file:
        data=json.load(file)
    
    with open(f"assets/token/matrices/{st.session_state['username']}.json","w") as file:
        data.append(new_matrix)
        json.dump(data,file)

        




def apply_and_display_algorithm(matrix, algorithm):

    if algorithm != AlgorithmType.NONE.value :
        if matrix is not None and st.session_state.input_type != InputType.CSV_UPLOAD.value:
            st.latex(matrix_to_latex(matrix))
        st.write(f"### Result of {algorithm}:")
        result, steps, descriptions = apply_algorithm(matrix, algorithm)
        if (st.session_state.input_type != InputType.CSV_UPLOAD.value):
            if result is not None:
                
                if isinstance(result, np.ndarray):
                    st.latex(matrix_to_latex(result))
                    download_csv(save_matrix_to_csv(result))
                    if "LOGGED_IN" in st.session_state and st.session_state["LOGGED_IN"]:
                         save_the_matrix(result,algorithm)
                else:
                    st.write(f"#### {result}")

                if algorithm == AlgorithmType.CHOLESKY.value and isinstance(result, np.ndarray):
                    st.write("### Steps of Cholesky Decomposition:")
                    for step, description in zip(steps, descriptions):
                        st.write(description)
                        st.latex(matrix_to_latex(step))
        elif (st.session_state.input_type == InputType.CSV_UPLOAD.value):
            if result is not None:
                if isinstance(result, np.ndarray):
                    download_csv(save_matrix_to_csv(result))
                    if "LOGGED_IN" in st.session_state and st.session_state["LOGGED_IN"]:
                        save_the_matrix(result,algorithm)

                else:
                    st.write(f"#### {result}")
    

def get_gemini_response(user_input):
    my_api_key = creds.Gemini_API
    genai.configure(api_key=my_api_key)
    try:
        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(user_input)

        return response.text.strip()

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI for the chatbot
def show_chatbot():
    with st.sidebar.expander("💬 Chat with MatX "):
        st.title("MatX Chatbot")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input23 = st.text_input("Ask me anything about algebra or the World:", "")
        if user_input23:
            response = get_gemini_response(user_input23)
            st.session_state.chat_history = [("User", user_input23), ("MatX", response)]

        for speaker, message in st.session_state.chat_history:
            if speaker == "User":
                st.write(f"**User**: {message}")
            else:
                st.write(f"**MatX**: {message}")
def cholesky_solve(matrix, algorithm):
    try:
        if algorithmes.isSquare(matrix) and algorithmes.isSymmetric(matrix) and algorithmes.is_positive_definite(matrix):
            b_matrix = handle_resolution(matrix)  # Get matrix B (for AX = B)
            if b_matrix is not None:
                solution = algorithmes.resolution(matrix, b_matrix)  # Solve AX = B
        else:
            st.write("### Matrix A must be square, symmetric, and positive definite")
            return
    except np.linalg.LinAlgError as e:
        st.write(f"Error in solving AX = B: {e}")
        return

    if st.button("Solve"):
        if b_matrix is None:
            st.write('### Upload vector b')
            return
        st.write("### Solution of AX = B:")
        
        # Only display the solution as LaTeX if the input type is not CSV
        if st.session_state.input_type != InputType.CSV_UPLOAD.value:
            if isinstance(solution, np.ndarray):
                st.latex(matrix_to_latex(solution))
                # Convert result to CSV and show download button
                csv_content = save_matrix_to_csv(solution)
                download_csv(csv_content)
                if "LOGGED_IN" in st.session_state and st.session_state["LOGGED_IN"]:
                    try:
                        save_the_matrix(solution, algorithm)
                    except:
                        pass
        else:
            # If input type is CSV, just display the result as text or table, not LaTeX
            if isinstance(solution, np.ndarray):
                st.write(f"#### Solution matrix X:")
                st.dataframe(solution)  # Display as a DataFrame table
                # Convert result to CSV and show download button
                csv_content = save_matrix_to_csv(solution)
                download_csv(csv_content)
                if "LOGGED_IN" in st.session_state and st.session_state["LOGGED_IN"]:
                    try:
                        save_the_matrix(solution, algorithm)
                    except:
                        pass
            else:
                st.write(f"#### {solution}")
          
def show_home():
    st.title("Matrix Operations")
    st.sidebar.header("Matrix Settings")

    # Session state initialization
    if "matrix" not in st.session_state:
        st.session_state.matrix = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = None
    if "algorithm" not in st.session_state:
        st.session_state.algorithm = None

    # Clear matrix button
    if st.sidebar.button("Clear Matrix"):
        clear_matrix()

    # Matrix input type selection
    type_of_input = [matrix_type.value for matrix_type in InputType]
    input_type = st.sidebar.radio("Select Matrix Input Type", type_of_input)

    if st.session_state.input_type != input_type:
        st.session_state.matrix = None
    st.session_state.input_type = input_type
    # Create zero matrix for L

    # Handle matrix input based on type
    handle_matrix_input(input_type)

    # Matrix operation selection
    st.sidebar.header("Matrix Operations")
    algorithm_type = [algorithm.value for algorithm in AlgorithmType]
    algorithm = st.sidebar.selectbox("Choose an Algorithm", algorithm_type)

    # Apply and display the selected algorithm result
    if st.session_state.matrix is not None and algorithm != AlgorithmType.CHOLESKY_RESOLUTION.value:
        apply_and_display_algorithm(st.session_state.matrix, algorithm)
    elif st.session_state.matrix is not None and algorithm == AlgorithmType.CHOLESKY_RESOLUTION.value:
        cholesky_solve(st.session_state.matrix,algorithm)
    show_chatbot()    
