const API_URL = 'http://localhost:8000/api/chat';

function scrollToBottom() {
    const chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
}

function createMessageElement(text, isUser = false) {
    const template = document.createElement('div');
    template.className = `message ${isUser ? 'user-message' : 'assistant-message'} entry-animation`;
    
    const avatarIcon = isUser ? '👤' : '🤖';
    
    template.innerHTML = `
        <div class="avatar">${avatarIcon}</div>
        <div class="bubble">${text}</div>
    `;
    
    return template;
}

function createTypingIndicator() {
    const template = document.createElement('div');
    template.className = 'message assistant-message entry-animation';
    template.id = 'typing-indicator-container';
    
    template.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble">
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    
    return template;
}

// Convert line breaks and basic markdown to HTML for the answer
function formatAnswer(text) {
    // Escape HTML first to prevent XSS
    let formatted = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Bold
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italics
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

async function sendMessage() {
    const inputField = document.getElementById('user-input');
    const question = inputField.value.trim();
    
    if (!question) return;
    
    const chatBox = document.getElementById('chat-box');
    
    // 1. Add User Message
    chatBox.appendChild(createMessageElement(question, true));
    inputField.value = '';
    scrollToBottom();
    
    // 2. Add Typing Indicator
    const typingIndicator = createTypingIndicator();
    chatBox.appendChild(typingIndicator);
    scrollToBottom();
    
    // 3. Send to API
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById('typing-indicator-container').remove();
        
        // Add Backend Answer
        chatBox.appendChild(createMessageElement(formatAnswer(data.answer)));
        scrollToBottom();
        
    } catch (error) {
        console.error("Failed to fetch response:", error);
        
        // Remove typing indicator
        document.getElementById('typing-indicator-container').remove();
        
        // Add Error Message
        const errorMsg = 'Sorry, there was an issue reaching the server. Ensure that the FastAPI backend is running on http://localhost:8000.';
        chatBox.appendChild(createMessageElement(errorMsg));
        scrollToBottom();
    }
}
