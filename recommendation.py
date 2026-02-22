import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class CounselorAI:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        """Initialize the AI with a Groq model."""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("CHAT_GROQ_API_KEY")

        # Initialize ChatGroq model
        self.chat_groq = ChatGroq(model_name=model_name, api_key=self.api_key)

        # Define the system prompt
        self.system_prompt = SystemMessage(
            content="You are a professional counselor. Your role is to provide expert guidance, support, and "
                    "personalized recommendations based on the user's conversation history. Your responses should "
                    "be empathetic, insightful, and helpful."
        )

    def load_chat_history(self, file_path):
        """Load chat history from a text file and format it into messages."""
        messages = []
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("You:"):
                    messages.append(HumanMessage(content=line.replace("You:", "").strip()))
                elif line.startswith("AI:"):
                    messages.append(AIMessage(content=line.replace("AI:", "").strip()))
        return messages

    def generate_recommendation(self, chat_history_file, user_id):
        """Generate personalized recommendations based on chat history."""
        # Load chat history
        chat_history = self.load_chat_history(chat_history_file)

        # Structured recommendation prompt
        recommendation_prompt = (
            "Based on my past conversations, provide a professional counseling recommendation. "
            "Ensure the response is well-structured with clear bullet points. Include practical exercises, "
            "self-help techniques, and additional resources (if applicable) to help address the user's concerns. "
            "Make the response empathetic, supportive, and easy to follow."
        )

        # Build message context and invoke model directly.
        messages = [self.system_prompt] + chat_history + [HumanMessage(content=recommendation_prompt)]
        response_obj = self.chat_groq.invoke(messages)
        response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

        # Save recommendation to file
        rec_dir = os.getenv("RECOMMENDATION_DIR", "recommendations")
        os.makedirs(rec_dir, exist_ok=True)
        rec_file_path = os.path.join(rec_dir, f"chat_{user_id}.txt")
        with open(rec_file_path, "w", encoding="utf-8") as f:
            f.write(response)

        return response

# Example Usage
if __name__ == "__main__":
    # Initialize AI Counselor
    counselor_ai = CounselorAI()

    # Provide chat history file path
    chat_history_file = r"chat_logs/chat_history_2.txt"

    # Get AI recommendation
    recommendation = counselor_ai.generate_recommendation(chat_history_file)

    # Print the recommendation
    print("AI Recommendation:\n", recommendation)
