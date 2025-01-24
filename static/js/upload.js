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
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const container = document.createElement('div');
            container.classList.add('row');

            const titleElement = document.createElement('div');
            titleElement.classList.add('title');
            titleElement.textContent = `${key}:`;

            const valueElement = document.createElement('div');
            valueElement.classList.add('value');
            valueElement.textContent = data[key] || 'N/A';

            container.appendChild(titleElement);
            container.appendChild(valueElement);
            result.appendChild(container);
        }
    }
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
        const response = await fetch('/extract_id/', {
            method: 'POST',
            body: formData,
        });
        const resultData = await response.json();
        displayAsTitleAndValue(resultData);
    } catch (error) {
        result.innerHTML = 'Error: ' + error.message;
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Extract Data';
    }
});
