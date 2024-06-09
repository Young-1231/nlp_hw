function processText() {
    const inputText = document.getElementById('inputText').value;
    window.pywebview.api.process_text(inputText).then(result => {
        document.getElementById('outputText').textContent = result;
    });
}
