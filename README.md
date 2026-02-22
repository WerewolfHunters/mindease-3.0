# MindEase 3.0

MindEase 3.0 is a Flask-based AI mental health counseling web app with:

- User authentication (signup/login/logout) using SQLite
- AI chat support powered by Groq via LangChain
- Recommendation generation from saved chat history
- Suicide-risk analysis with email alerting for high-risk cases
- Unified dashboard UI for Chat, Video, and Recommendation tabs

## Project Structure

- `app.py`: Main Flask app, routes, auth, chat/recommendation/suicide pipeline integration
- `conversation.py`: Chatbot logic using `ChatGroq` + file-based history
- `recommendation.py`: Recommendation generation pipeline
- `RAGclassifier.py`: FAISS-based similarity classifier used for risk/label analysis
- `suicide_detector.py`: Email alert sender for suicide-risk triggers
- `templates/`: Jinja templates for landing/auth/dashboard pages
- `static/`: CSS/JS/assets
- `users.db`: SQLite user database (local runtime)
- `chat_logs/`: Saved chat history files (runtime)
- `recommendations/`: Generated recommendation files (runtime)

## Requirements

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create your local `.env` from `.env.example`:

```bash
cp .env.example .env
```

Required values:

- `CHAT_GROQ_API_KEY`: Groq API key for chat and recommendation generation
- `MAIL`: Sender email address used for suicide alert emails
- `PASS`: Sender email app password (recommended for Gmail)

## Run the App

```bash
python app.py
```

Default local URL:

- `http://127.0.0.1:5000`

## Deploy on Vercel

This repository is configured for Vercel serverless deployment via `vercel.json`.

### 1. Push to GitHub

Push your project to a GitHub repository.

### 2. Import Project in Vercel

- Create a new project in Vercel
- Import your GitHub repository

### 3. Set Environment Variables in Vercel

Add these variables in Vercel Project Settings:

- `CHAT_GROQ_API_KEY`
- `MAIL`
- `PASS`
- Optional: `DB_PATH` (defaults to `/tmp/users.db` on Vercel runtime)

### 4. Deploy

Vercel will use:
- `vercel.json` routing/build config
- `requirements.txt` for Python dependencies

### Important Vercel Notes

- Vercel serverless filesystem is ephemeral:
  - `chat_logs/`, `recommendations/`, and SQLite DB are runtime-local (`/tmp`) and not persistent across cold starts.
- For production persistence, migrate to:
  - managed database (Postgres/MySQL/Supabase/etc.)
  - external storage for chat/recommendation files
- Heavy ML dependencies are marked optional in `requirements.txt` to keep deployment lightweight.

## Core User Flow

1. User signs up on `/signup` (saved in `users.db`)
2. User logs in on `/login`
3. User chats on `/dashboard`
4. On `End Chat`:
   - chat is saved
   - recommendation pipeline runs
   - suicide analysis runs
   - if high risk is triggered, alert email is sent to the registered signup email
   - user is redirected to recommendation tab

## Notes

- `.env`, `.db`, `chat_logs/`, and `recommendations/` are excluded in `.gitignore`
- `dataset/` and large model artifacts are excluded by default to keep repository lightweight
- If email alerts fail, check SMTP/app-password setup and terminal debug logs
