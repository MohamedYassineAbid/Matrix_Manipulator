function tableToMatrix(table) {
    const rows = table.rows.length;
    if (rows === 0) return [];
    
    const cols = table.rows[0].cells.length;
    const matrix = [];
    
    for (let i = 0; i < rows; i++) {
        matrix[i] = [];
        for (let j = 0; j < cols; j++) {
            matrix[i][j] = document.getElementById(`${table.id}[${i}][${j}]`).value;
        }
    }
    
    return matrix;
}

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
            html += `<td><input type="text" id="${table.id}[${i}][${j}]" value="${matrix[i][j]}" class="matrix__item form__input"/></td>`;
        }
        html += "</tr>";
    }
    
    table.innerHTML = html;
}

function resizeMatrix(originalMatrix, newRows, newCols) {
    const resizedMatrix = [];
    
    for (let i = 0; i < newRows; i++) {
        resizedMatrix[i] = [];
        for (let j = 0; j < newCols; j++) {
            if (i < originalMatrix.length && j < originalMatrix[i].length) {
                resizedMatrix[i][j] = originalMatrix[i][j];
            } else {
                resizedMatrix[i][j] = ""; // Fill with empty string if out of bounds
            }
        }
    }
    
    return resizedMatrix;
}

document.getElementById("resize-btn").addEventListener("click", function () {
    const rows = Math.max(1, Math.min(10, parseInt(document.getElementById("rows").value))); // Limit rows to 1-10
    const cols = Math.max(1, Math.min(10, parseInt(document.getElementById("cols").value))); // Limit columns to 1-10
    
    const table = document.getElementById("matrix-table");
    const currentMatrix = tableToMatrix(table);
    const resizedMatrix = resizeMatrix(currentMatrix, rows, cols);
    matrixToTable(resizedMatrix, table);
    
    // Render the matrix using KaTeX
    renderMatrix(resizedMatrix);
});

// Function to render matrix with KaTeX
function renderMatrix(matrix) {
    const latex = matrixToLatex(matrix);
    const container = document.getElementById('matrix-render');
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
matrixToTable([["", "", ""], ["", "", ""], ["", "", ""]], document.getElementById("matrix-table"));
