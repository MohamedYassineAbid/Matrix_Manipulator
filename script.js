// Convert table to matrix
function tableToMatrix(table) {
    const rows = table.rows.length;
    if (rows === 0) return [];
    
    const cols = table.rows[0].cells.length;
    const matrix = [];
    
    for (let i = 0; i < rows; i++) {
        matrix[i] = [];
        for (let j = 0; j < cols; j++) {
            const cell = table.rows[i].cells[j];
            matrix[i][j] = cell.dataset.value || "0";  // Store matrix value in cell's data-value
        }
    }
    
    return matrix;
}

// Convert matrix to table (clickable for input)
function matrixToTable(matrix, table) {
    if (matrix.length === 0) {
        table.innerHTML = "";
        return;
    }

    const rows = matrix.length;
    const cols = matrix[0].length;
    let html = "";

    for (let i = 0; i < rows; i++) {
        html += "<tr>";
        for (let j = 0; j < cols; j++) {
            const value = matrix[i][j] || "0";
            html += `<td data-value="${value}" id="${table.id}[${i}][${j}]">${renderLatex(value)}</td>`;
        }
        html += "</tr>";
    }
    
    table.innerHTML = html;

    // Add event listeners to allow editing
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cellId = `${table.id}[${i}][${j}]`;
            const cell = document.getElementById(cellId);
            cell.addEventListener('click', function () {
                createInputForCell(cell);
            });
        }
    }
}

// Create input field when a cell is clicked
function createInputForCell(cell) {
    const currentValue = cell.dataset.value || "0";
    const input = document.createElement("input");
    input.type = "text";
    input.value = currentValue;
    input.classList.add("matrix-input");

    input.addEventListener("blur", function () {
        const newValue = input.value;
        cell.dataset.value = newValue;
        cell.innerHTML = renderLatex(newValue);
    });

    cell.innerHTML = "";
    cell.appendChild(input);
    input.focus();
}

// Render a matrix cell value using KaTeX
function renderLatex(value) {
    try {
        return katex.renderToString(value);
    } catch (error) {
        return value;  // Fallback to raw value if KaTeX rendering fails
    }
}

// Resize the matrix
function resizeMatrix(originalMatrix, newRows, newCols) {
    const resizedMatrix = [];
    
    for (let i = 0; i < newRows; i++) {
        resizedMatrix[i] = [];
        for (let j = 0; j < newCols; j++) {
            if (i < originalMatrix.length && j < originalMatrix[i].length) {
                resizedMatrix[i][j] = originalMatrix[i][j];
            } else {
                resizedMatrix[i][j] = "0"; // Default new cells to "0"
            }
        }
    }
    
    return resizedMatrix;
}

// Resize button event listener
document.getElementById("resize-btn").addEventListener("click", function () {
    const rows = parseInt(document.getElementById("rows").value);
    const cols = parseInt(document.getElementById("cols").value);

    if (rows > 10 || cols > 10) {
        alert("You cannot create more than 10 rows or columns. Please upload a file for larger matrices.");
        document.getElementById("file-input").click(); // Trigger file input
        return; // Stop execution if the limit is exceeded
    }

    const table = document.getElementById("matrix-table");
    const currentMatrix = tableToMatrix(table);
    const resizedMatrix = resizeMatrix(currentMatrix, Math.max(1, Math.min(10, rows)), Math.max(1, Math.min(10, cols)));
    matrixToTable(resizedMatrix, table);
    
    // Render the matrix using KaTeX
    renderMatrix(resizedMatrix);
});

// Handle file upload
document.getElementById("upload-btn").addEventListener("click", function () {
    document.getElementById("file-input").click(); // Open file dialog
});

// Read the uploaded file and convert it to a matrix
document.getElementById("file-input").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const content = e.target.result;
            const matrix = parseMatrixFile(content); // Function to parse matrix from file
            matrixToTable(matrix, document.getElementById("matrix-table"));
            renderMatrix(matrix); // Render the matrix
        };
        reader.readAsText(file);
    }
});

// Function to parse matrix from file content (assumes rows are separated by newlines and values by spaces)
function parseMatrixFile(content) {
    const rows = content.trim().split("\n");
    return rows.map(row => row.split(/\s+/)); // Split by whitespace
}

// Function to render matrix with KaTeX in a separate area
function renderMatrix(matrix) {
    const latex = matrixToLatex(matrix);
    const container = document.getElementById('matrix-render');
    container.innerHTML = ''; // Clear previous renders
    katex.render(latex, container);
}

// Function to convert matrix into LaTeX format for KaTeX
function matrixToLatex(matrix) {
    let latex = '\\begin{bmatrix}';
    for (let i = 0; i < matrix.length; i++) {
        latex += matrix[i].join(' & ');
        if (i < matrix.length - 1) {
            latex += ' \\\\ ';
        }
    }
    latex += '\\end{bmatrix}';
    return latex;
}

// Initialize the matrix on page load
document.addEventListener("DOMContentLoaded", function() {
    matrixToTable([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]], document.getElementById("matrix-table"));
});
