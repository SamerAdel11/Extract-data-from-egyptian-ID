const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const submitButton = document.getElementById('submitButton');
const result = document.getElementById('result');

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        submitButton.style.display = 'inline-block';
        result.innerHTML = '';
    } else {
        submitButton.style.display = 'none';
    }
});

function displayAsTitleAndValue(data) {
    result.innerHTML = '';

    // Create the table element
    const table = document.createElement('table');
    table.classList.add('data-table'); // Add a class for styling if needed

    // Loop through the data to create table rows
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const row = document.createElement('tr'); // Create a table row

            // Create a cell for the title (key)
            const titleCell = document.createElement('td');
            titleCell.classList.add('title');
            titleCell.textContent = `${key}:`;

            // Create a cell for the value
            const valueCell = document.createElement('td');
            valueCell.classList.add('value');
            valueCell.textContent = data[key] || 'N/A';

            // Append the cells to the row
            row.appendChild(titleCell);
            row.appendChild(valueCell);

            // Append the row to the table
            table.appendChild(row);
        }
    }

    // Append the table to the result container
    result.appendChild(table);
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    submitButton.disabled = true;
    submitButton.textContent = 'Please wait...';

    try {
        const response = await fetch('/extract_id', {
            method: 'POST',
            body: formData,
        });
        const resultData = await response.json();
        displayAsTitleAndValue(resultData);
        if (resultData['Id'].length<14) {
            throw new Error("ID number is less than 14");
        }
    } catch (error) {
        result.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Extract Data';
    }
});
