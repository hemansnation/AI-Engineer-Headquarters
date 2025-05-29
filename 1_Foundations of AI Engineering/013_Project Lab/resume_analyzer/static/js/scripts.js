document.querySelector('form').addEventListener('submit', function(e) {
    const fileInput = document.querySelector('#resume');
    if (!fileInput.value) {
        e.preventDefault();
        alert('Please select a PDF file.');
    }
});