import streamlit as st


def show_user_guide():
    st.write(
        """
        # Matrix Manipulator User Guide

## **Introduction**
This application allows users to perform a variety of matrix operations, including generating matrices, applying algorithms, and visualizing the results. Users can interact with the application through a simple web interface built using Streamlit.

## **Key Features**
1. **Matrix Input Types**
    - **Manual Input**: Users can manually input matrix values row by row.
    - **Random Generation**: Users can generate random matrices based on selected properties.
    - **CSV Upload**: Users can upload a CSV file containing matrix data.
    
2. **Matrix Types**
    - **Square**: A matrix with an equal number of rows and columns.
    - **Symmetric**: A matrix that is equal to its transpose.
    - **Diagonal**: A matrix where non-diagonal elements are zero.
    - **Band**: A matrix with only a narrow band around the diagonal containing non-zero elements.
    - **Identity**: A matrix with ones on the diagonal and zeros elsewhere.

3. **Matrix Operations**
    - **Transpose**: Flips the matrix along its diagonal.
    - **Determinant**: Computes the determinant of a square matrix.
    - **Inverse**: Computes the inverse of a square matrix, if possible.
    - **Cholesky Decomposition**: Decomposes a positive-definite matrix into a lower triangular matrix.
    - **Positivity Check**: Checks if the matrix is positive definite.

4. **Matrix Visualization**
    - **LaTeX Display**: Displays matrices in LaTeX format for better readability.
    - **CSV Export**: Users can download processed matrices as CSV files.

5. **Chatbot Integration**
    - **MatX Chatbot**: Ask the chatbot about matrices and algebra. It uses the Google Gemini AI model to answer queries.

---

## **How to Use**

### 1. **Matrix Input**
- **Manual Input**: Select "Manual Input" from the sidebar, then specify the number of rows and columns. Enter matrix values in the input fields.
- **Random Generation**: Select "Random" and choose the matrix type (e.g., Square, Symmetric) and matrix size. You can also specify the range of values for matrix elements.
- **CSV Upload**: Select "CSV Upload" and upload a CSV file. The matrix will be displayed, and operations can be applied.

### 2. **Matrix Operations**
- After entering or uploading a matrix, choose an operation from the **Matrix Operations** section in the sidebar:
    - **Transpose**: Flips the matrix across its diagonal.
    - **Determinant**: Calculates the determinant for square matrices.
    - **Inverse**: Computes the inverse of square matrices.
    - **Cholesky Decomposition**: Decomposes symmetric positive-definite matrices.
    - **Positivity Check**: Checks if the matrix is positive-definite.

    Results will be displayed in LaTeX format or as a message.

### 3. **Matrix Visualization**
- The matrix will be displayed in LaTeX format after you input or upload it.
- After applying an operation, the result will also be displayed in LaTeX format.
- If a matrix operation generates a new matrix, you can download the result as a CSV file by clicking the **Download Processed Matrix** button.

### 4. **Chatbot**
- You can ask questions related to matrices and algebra through the chatbot in the sidebar. Simply type a question and receive a response from the MatX chatbot powered by Google Gemini.

---

## **Matrix Types and Operations**

### **Matrix Types**
- **Square Matrix**: A matrix with the same number of rows and columns.
- **Symmetric Matrix**: A matrix that is equal to its transpose.
- **Diagonal Matrix**: A matrix where all non-diagonal elements are zero.
- **Band Matrix**: A matrix with non-zero elements only in a narrow band around the diagonal.
- **Identity Matrix**: A square matrix with ones on the diagonal and zeros elsewhere.

### **Operations**
- **Transpose**: Flips the rows and columns of the matrix.
- **Determinant**: A scalar value that can be computed only for square matrices. It indicates the matrix's singularity and invertibility.
- **Inverse**: The inverse matrix, if it exists, is the matrix that, when multiplied by the original matrix, results in the identity matrix.
- **Cholesky Decomposition**: Decomposes a positive-definite matrix into a lower triangular matrix.
- **Positivity Check**: Checks if the matrix is positive-definite, which is a requirement for certain decompositions like Cholesky.

---

## **Conclusion**
This Matrix Manipulator app offers a simple and effective way to handle matrix operations and visualizations. Whether you're working with small matrices or large complex ones, the app provides the tools you need for efficient matrix manipulation.

        """
    )
