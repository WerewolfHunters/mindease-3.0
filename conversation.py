import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage


class CounselorChatbot:
    def __init__(self, model_name="llama-3.1-8b-instant", chat_directory="chat_logs"):
        """Initialize the AI chatbot with file-based chat history."""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("CHAT_GROQ_API_KEY")

        # Initialize ChatGroq model
        self.chat_groq = ChatGroq(model_name=model_name, api_key=self.api_key)

        # System prompt defining the AI's role
        self.system_prompt = SystemMessage(
            content="You are a friendly and professional counselor. Your role is to provide concise, supportive, "
                    "and insightful guidance based on the user's conversation history. Keep responses short, "
                    "empathetic, and helpful."
        )

        # Directory for storing chat logs
        self.chat_directory = chat_directory
        os.makedirs(self.chat_directory, exist_ok=True)

    def get_chat_history_path(self, user_id):
        """Generate the chat history file path for a given user."""
        return os.path.join(self.chat_directory, f"chat_history_{user_id}.txt")

    def load_chat_history(self, user_id):
        """Load previous chat history from a file if available."""
        chat_history_file = self.get_chat_history_path(user_id)
        messages = []

        if os.path.exists(chat_history_file):
            with open(chat_history_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("You:"):
                        messages.append(HumanMessage(content=line.replace("You:", "").strip()))
                    elif line.startswith("AI:"):
                        messages.append(AIMessage(content=line.replace("AI:", "").strip()))
        return messages

    def save_chat_history(self, user_id, chat_history):
        """Save the chat history to a text file."""
        chat_history_file = self.get_chat_history_path(user_id)

        with open(chat_history_file, "w", encoding="utf-8") as file:
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    file.write(f"You: {message.content}\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n")

    def chat(self, user_id, user_input):
        """Generate AI response for the given user input and update chat history."""
        previous_chat_history = self.load_chat_history(user_id)

        if not self.api_key:
            raise ValueError("CHAT_GROQ_API_KEY is missing in environment variables.")

        messages = [self.system_prompt] + previous_chat_history + [HumanMessage(content=user_input)]
        response = self.chat_groq.invoke(messages)
        ai_response = response.content if hasattr(response, "content") else str(response)

        # Update and save chat history
        previous_chat_history.append(HumanMessage(content=user_input))
        previous_chat_history.append(AIMessage(content=ai_response))
        self.save_chat_history(user_id, previous_chat_history)

        return ai_response

    def clear_memory(self, user_id):
        """No-op for compatibility. Chat history is file-based."""
        return None


if __name__ == "__main__":
    chatbot = CounselorChatbot()
    print(chatbot.api_key)
    user_id = input("Enter your User ID: ")

    print("🔹 AI Counselor: Hello! I'm here to listen and support you. Type 'exit' to end the session.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "end chat"]:
            print("🔹 AI Counselor: It was great talking to you. Take care! 😊")
            break

        ai_response = chatbot.chat(user_id, user_input)
        print(f"🔹 AI Counselor: {ai_response}")
