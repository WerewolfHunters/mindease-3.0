from conversation import CounselorChatbot
from recommendation import CounselorAI


def main():
    user_id = input("Enter your User ID: ")  # Unique user ID for chat history tracking

    # Start chat session
    chatbot = CounselorChatbot()
    chatbot.chat(user_id)

    # Generate recommendation based on chat history
    print("\n🔹 Generating your personalized recommendation...\n")
    counselor_ai = CounselorAI()
    chat_history_file = chatbot.get_chat_history_path(user_id)
    recommendation = counselor_ai.generate_recommendation(chat_history_file)

    # Display recommendation
    print("🔹 AI Recommendation:\n", recommendation)


if __name__ == "__main__":
    main()
