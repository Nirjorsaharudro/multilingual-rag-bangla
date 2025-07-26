const chatArea = document.getElementById("chatArea");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");

let socket = null;

// Initialize WebSocket connection
function initWebSocket() {
    socket = new WebSocket("ws://localhost:8000/ws/chat");

    socket.onopen = () => {
        console.log("WebSocket connected");
        appendMessage("system", "Connected to chat server.");
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        appendMessage(msg.role, msg.content);
    };

    socket.onclose = () => {
        console.log("Socket closed");
        appendMessage("system", "Connection closed. Please refresh to reconnect.");
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        appendMessage("system", "An error occurred. Please try again.");
    };
}

// Append a message to the chat area
function appendMessage(role, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const htmlContent = DOMPurify.sanitize(marked.parse(content));
    messageDiv.innerHTML = htmlContent;

    chatArea.appendChild(messageDiv);
    scrollToBottom();
}

// Scroll to the latest message
function scrollToBottom() {
    chatArea.scrollTo({
        top: chatArea.scrollHeight,
        behavior: "smooth",
    });
}

// Send a message
function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || !socket || socket.readyState !== WebSocket.OPEN) return;

    const userMsg = { role: "user", content };
    appendMessage("user", content);
    socket.send(JSON.stringify(userMsg));
    messageInput.value = "";
}

// Event listeners
sendButton.addEventListener("click", sendMessage);
messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Initialize WebSocket when the page loads
window.addEventListener("load", initWebSocket);
