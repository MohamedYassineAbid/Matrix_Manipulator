
# Matrix Manipulator

Matrix Manipulator is a web application built with Python and Streamlit that enables users to perform various matrix operations. The app supports user login, matrix generation, manipulation, and allows users to view their operation history.
[The online Website](https://matrixmanipulator.streamlit.app/)
## Features

- **User Authentication**: Login system for personalized sessions.
- **Matrix Generation**: Randomly generate different types of matrices for testing and learning.
- **Matrix Operations**: Includes basic and advanced matrix manipulations.
- **History Tracking**: Maintains a history of user operations.

## Prerequisites

- **Python 3.8 or higher**
- **Poetry** - for dependency management. Install Poetry following the instructions in [Poetry's documentation](https://python-poetry.org/docs/#installation).

## Setup Instructions

### Step 1: Clone the Repository

1. Open a terminal window.
2. Run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/MohamedAliJmal/Matrix_Manipulator
   ```
3. Navigate into the project directory:
   ```bash
   cd Matrix_Manipulator
   ```

### Step 2: Install Poetry

If Poetry is not already installed, follow these steps to install it:

1. Run the following command to download and install Poetry:
   ```bash
   sudo apt install python3-pip -y
   ```
    ```bash
   pip install poetry
   ```
2. Add Poetry to your PATH if necessary by following [these instructions](https://python-poetry.org/docs/#installation).

### Step 3: Install Dependencies

1. Once inside the project directory, use Poetry to set up a virtual environment and install all dependencies specified in `pyproject.toml`:
   ```bash
   poetry shell
   ```

   ```bash
   poetry install
   ```
2. Set Poetry venv as Python Interpreter :
   ```bash
   poetry env info --path
   ```
      - Open your project in VS Code.
      - Press Ctrl+Shift+P (or Cmd+Shift+P on macOS) to open the Command Palette.
      - Search for and select Python: Select Interpreter.
      - Select #Enter #Intrepreter #Path
      - Paste  the command output
### Step 4: Run the Application

1. After installation, start the application by running:
   ```bash
   streamlit run main.py
   ```
2. This command will launch the application in your default web browser. If it doesn’t open automatically, go to [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. **Login**: Log in to your account.
2. **Generate Matrix**: Use the matrix generator to create various types of matrices.
3. **Perform Operations**: Choose from a range of operations available in the app.
4. **View History**: Review your operation history for previous manipulations.


## License

This project is licensed under the MIT License.
