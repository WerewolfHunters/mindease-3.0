const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");

function addMessage(text, sender) {
    const row = document.createElement("div");
    row.className = `chat-msg ${sender}`;
    const bubble = document.createElement("p");
    bubble.textContent = text;
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage(message) {
    const response = await fetch("/get_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: message })
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || "Failed to get response.");
    }
    return data.response;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    chatInput.value = "";
    addMessage("Thinking...", "bot");

    try {
        const reply = await sendMessage(message);
        const pending = chatMessages.querySelector(".chat-msg.bot:last-child p");
        if (pending && pending.textContent === "Thinking...") {
            pending.textContent = reply;
        } else {
            addMessage(reply, "bot");
        }
    } catch (error) {
        const pending = chatMessages.querySelector(".chat-msg.bot:last-child p");
        if (pending && pending.textContent === "Thinking...") {
            pending.textContent = "Unable to respond right now. Please try again.";
        } else {
            addMessage("Unable to respond right now. Please try again.", "bot");
        }
    }
});
