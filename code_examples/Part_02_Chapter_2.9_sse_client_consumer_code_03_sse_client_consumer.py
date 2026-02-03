const eventSource = new EventSource('/ask/stream?question=' + encodeURIComponent(userQuestion));
const answerDiv = document.getElementById('answer');

eventSource.onmessage = (event) => {
    // Append each token to the display
    answerDiv.textContent += event.data;
};

eventSource.onerror = (error) => {
    console.error('Stream error:', error);
    eventSource.close();
};

// Auto-close when complete (server ends stream)
eventSource.addEventListener('close', () => {
    eventSource.close();
});
