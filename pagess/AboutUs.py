import streamlit as st

def show_examples():
    st.header("About Us")
    st.write(
        """
        **Matrix Manipulator** is an academic project designed to provide students, educators, and researchers with an accessible and user-friendly platform for performing matrix operations. Our goal is to simplify complex matrix calculations and offer interactive learning experiences for those studying linear algebra and related fields.

        ---

        ### Project Overview

        The **Matrix Manipulator** app allows users to:
        - Perform a variety of matrix operations, such as **Transpose**, **Determinant**, **Inverse**, and **Cholesky Decomposition**.
        - View and manipulate matrices in LaTeX format for clarity and ease of use.
        - Input matrices manually, generate them randomly, or upload them via CSV files.
        - Interact with an AI-powered chatbot (**MatX**) for assistance with algebra and matrix-related concepts.
        - Download processed matrices as CSV files for further analysis.

        This project is designed to support learning and research in fields that heavily rely on matrix mathematics, such as data science, machine learning, and engineering.

        ---
""")
    st.header("Meet the Team")
    # Display supervisor's information with an image
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/supervisor_picture.jpg", width=100)
    with col2:
        st.markdown("###  Dr. Sirine Marrakchi")
        st.write("Data Engineering Student")
        st.write("Faculty of Sciences of Sfax")
    
    # Display team members' information with their images
    st.write("---")
    
    # Add your team member pictures and details
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/abayda.png", width=100)
        
    
    with col2:
        st.markdown("#### [Mohamed Yassine Abid](https://github.com/MohamedYassineAbid)")
        st.write("Data Engineering Student")
        st.write("Faculty of Sciences of Sfax")
    
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/jmal.jpeg", width=100)
    with col2:
        st.markdown("#### [Mohamed Ali Djmal](https://github.com/chameauu)")
        st.write("Data Engineering Student")
        st.write("Faculty of Sciences of Sfax")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/chakster.png", width=100)
    with col2:
        st.markdown("#### [Ahmad Chakcha](https://github.com/AhmadChakcha)")
        st.write("Data Engineering Student")
        st.write("Faculty of Sciences of Sfax")
        
    st.write("""
            We welcome your feedback and suggestions! If you have any questions, would like to report an issue, or want to learn more about the project, please reach out to us:

            - **Email**: manipulatormatrix@gmail.com 
            - **GitHub**: [Matrix Manipulator](https://github.com/MohamedAliJmal/Matrix_Manipulator)
            """
        )