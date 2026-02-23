from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from conversation import CounselorChatbot
from recommendation import CounselorAI
from suicide_detector import MentalHealthMonitor
from dotenv import load_dotenv
import sqlite3
import json
import uuid  # <-- for generating unique session_id
import os
import traceback
import re
import requests

load_dotenv()

try:
    from RAGclassifier import RAGSimilarityClassifier
except Exception as import_error:
    RAGSimilarityClassifier = None
    print(f"[startup] RAG classifier import skipped: {import_error}")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
embedding_path = './model/embeddings.npy'
FALLBACK_DATASET_PATH = './model/balanced_cleaned_dataset.csv'
DATASET_FILE_ID = os.getenv("SUICIDE_DATASET_FILE_ID", "1XwfATug0EkZMYWNwqdalQdvWdUF7yJI6")
DATASET_GDRIVE_URL = os.getenv(
    "SUICIDE_DATASET_URL",
    f"https://drive.google.com/uc?export=download&id={DATASET_FILE_ID}"
)
IS_VERCEL = os.getenv("VERCEL") == "1"
IS_RENDER = os.getenv("RENDER") == "true"
BASE_DATA_DIR = "/tmp" if (IS_VERCEL or IS_RENDER) else "."
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DATA_DIR, "users.db"))
DATASET_CACHE_PATH = os.path.join(BASE_DATA_DIR, "model", "balanced_cleaned_data.csv")

STRONG_RISK_PATTERNS = [
    r"\bi\s*(want|wanna)\s*to\s*die\b",
    r"\bi\s*(do\s*not|don't|dont)\s*want\s*to\s*live\b",
    r"\bi\s*(can't|cant)\s*go\s*on\b",
    r"\bkill\s*myself\b",
    r"\bend\s*my\s*life\b",
    r"\btake\s*my\s*own\s*life\b",
    r"\bno\s*reason\s*to\s*live\b",
    r"\bsuicid(e|al)\b",
    r"\bself\s*harm\b",
    r"\bharm\s*myself\b",
    r"\boverdos(e|ing)\b",
    r"\bhang\s*myself\b",
    r"\bjump\s*off\b"
]

MEDIUM_RISK_PATTERNS = [
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\bempty\b",
    r"\balone\b",
    r"\bdepressed\b",
    r"\blife\s*is\s*pointless\b",
    r"\bgive\s*up\b",
    r"\bwant\s*to\s*disappear\b",
    r"\bbetter\s*off\s*dead\b",
    r"\bnot\s*worth\s*living\b"
]

def get_chat_dir():
    return os.path.join(BASE_DATA_DIR, "chat_logs")

def get_recommendation_dir():
    return os.path.join(BASE_DATA_DIR, "recommendations")

def get_chat_file(user_id: str):
    return os.path.join(get_chat_dir(), f"chat_history_{user_id}.txt")

def get_recommendation_file(user_id: str):
    return os.path.join(get_recommendation_dir(), f"chat_{user_id}.txt")

def _download_from_gdrive(file_url: str, destination: str):
    session = requests.Session()
    response = session.get(file_url, stream=True, timeout=120)
    response.raise_for_status()

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(f"{file_url}&confirm={token}", stream=True, timeout=120)
        response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def get_dataset_path():
    """
    Returns dataset path with Google Drive as primary source and local file as fallback.
    """
    os.makedirs(os.path.dirname(DATASET_CACHE_PATH), exist_ok=True)

    if os.path.exists(DATASET_CACHE_PATH) and os.path.getsize(DATASET_CACHE_PATH) > 0:
        return DATASET_CACHE_PATH

    try:
        print(f"[dataset] Downloading dataset from Google Drive -> {DATASET_CACHE_PATH}")
        _download_from_gdrive(DATASET_GDRIVE_URL, DATASET_CACHE_PATH)
        print(f"[dataset] Downloaded successfully: {DATASET_CACHE_PATH}")
        return DATASET_CACHE_PATH
    except Exception as e:
        print(f"[dataset] Download failed: {e}")
        if os.path.exists(FALLBACK_DATASET_PATH):
            print(f"[dataset] Falling back to local dataset: {FALLBACK_DATASET_PATH}")
            return FALLBACK_DATASET_PATH
        return None

sender_mail = os.getenv("MAIL")
sender_pass = os.getenv("PASS")
os.environ["RECOMMENDATION_DIR"] = get_recommendation_dir()


chatbot = None
counselor_ai = None
detector = MentalHealthMonitor(sender_email=sender_mail, sender_password=sender_pass)
print(f"[startup] runtime={'vercel' if IS_VERCEL else 'render' if IS_RENDER else 'local'}")
print(f"[startup] BASE_DATA_DIR={BASE_DATA_DIR}")
print(f"[startup] DB_PATH={DB_PATH}")
print(f"[startup] CHAT_DIR={get_chat_dir()}")
print(f"[startup] REC_DIR={get_recommendation_dir()}")

def get_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = CounselorChatbot(chat_directory=get_chat_dir())
    return chatbot

def get_counselor_ai():
    global counselor_ai
    if counselor_ai is None:
        counselor_ai = CounselorAI()
    return counselor_ai

# DB Initialization
def init_db():
    os.makedirs(get_chat_dir(), exist_ok=True)
    os.makedirs(get_recommendation_dir(), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                mobile TEXT NOT NULL,
                userid TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mental_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userid TEXT NOT NULL,
                score INTEGER NOT NULL,
                label_counts TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def get_registered_email(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM users WHERE userid=?', (user_id,))
        row = cursor.fetchone()
        return row[0] if row else None

def extract_user_messages(chat_file: str):
    messages = []
    with open(chat_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("You:"):
                messages.append(line[4:].strip())
    return messages

def keyword_based_suicide_labels(chat_file: str):
    """
    Regex + keyword fallback detector used when FAISS-based classifier cannot run.
    Returns label_counts compatible with existing detector.evaluate_and_notify().
    """
    user_msgs = extract_user_messages(chat_file)
    if not user_msgs:
        return {"normal": 1}

    suicide_hits = 0
    total = len(user_msgs)

    for idx, msg in enumerate(user_msgs, start=1):
        text = re.sub(r"\s+", " ", msg.lower()).strip()
        strong_matches = [p for p in STRONG_RISK_PATTERNS if re.search(p, text)]
        medium_matches = [p for p in MEDIUM_RISK_PATTERNS if re.search(p, text)]

        if strong_matches or len(medium_matches) >= 2:
            suicide_hits += 1
            print(
                f"[suicide_fallback] risk message idx={idx}, "
                f"strong_matches={len(strong_matches)}, medium_matches={len(medium_matches)}"
            )
        else:
            print(
                f"[suicide_fallback] normal message idx={idx}, "
                f"strong_matches={len(strong_matches)}, medium_matches={len(medium_matches)}"
            )

    normal_hits = max(total - suicide_hits, 0)
    print(f"[suicide_fallback] label_counts={{'suicide': {suicide_hits}, 'normal': {normal_hits}}}")
    return {"suicide": suicide_hits, "normal": normal_hits}

def analyze_suicide_and_notify(user_id: str):
    """Analyze chat history for suicide risk and send alert email if threshold is crossed."""
    chat_file = get_chat_file(user_id)

    if not os.path.exists(chat_file):
        print(f"[suicide_detector] Skipped: chat file not found for user_id={user_id} -> {chat_file}")
        return {"action_taken": False, "suicide_percentage": None}

    try:
        # Try FAISS/RAG-based classifier first.
        # If it fails due to memory/runtime constraints, fallback to lightweight keyword model.
        try:
            if RAGSimilarityClassifier is None:
                raise RuntimeError("RAG classifier unavailable in this runtime.")
            dataset_path = get_dataset_path()
            if not dataset_path:
                raise RuntimeError("Dataset unavailable from Google Drive and no local fallback found.")
            classifier = RAGSimilarityClassifier(dataset_path, embedding_path, filepath=chat_file)
            _, label_counts = classifier.predict_labels()
            print(f"[suicide_detector] Using RAG classifier for user_id={user_id}")
        except MemoryError:
            print(f"[suicide_detector] RAG classifier MemoryError (std::bad_alloc). Falling back to keyword detector for user_id={user_id}")
            label_counts = keyword_based_suicide_labels(chat_file)
        except Exception as rag_error:
            print(f"[suicide_detector] RAG classifier failed: {rag_error}. Falling back to keyword detector for user_id={user_id}")
            label_counts = keyword_based_suicide_labels(chat_file)

        total = sum(label_counts.values())
        suicide_count = label_counts.get('suicide', 0)

        if total == 0:
            print(f"[suicide_detector] Skipped: no labels to evaluate for user_id={user_id}")
            return {"action_taken": False, "suicide_percentage": 0}

        percentage = (suicide_count / total) * 100
        print(f"[suicide_detector] user_id={user_id}, suicide_percentage={percentage:.2f}%")

        if percentage >= 3:
            user_mail = get_registered_email(user_id) or session.get("user_mail")
            if not user_mail:
                print(f"[suicide_detector] TRIGGERED but no registered email found for user_id={user_id}")
                return {"action_taken": False, "suicide_percentage": percentage}

            print(f"[suicide_detector] TRIGGERED for user_id={user_id}. Attempting alert to {user_mail}")
            email_sent = detector.evaluate_and_notify(label_counts=label_counts, user_email=user_mail)
            print(f"[suicide_detector] Email sent status for user_id={user_id}: {email_sent}")
            return {"action_taken": bool(email_sent), "suicide_percentage": percentage}

        print(f"[suicide_detector] NOT TRIGGERED for user_id={user_id}. No email sent.")
        return {"action_taken": False, "suicide_percentage": percentage}
    except Exception as e:
        print(f"[suicide_detector] Error during analysis for user_id={user_id}: {e}")
        traceback.print_exc()
        return {"action_taken": False, "suicide_percentage": None}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if "user_id" not in session:
        flash("Please login to continue.", "error")
        return redirect(url_for("login"))
    print(f"[route] /dashboard loaded for user_id={session.get('user_id')}")
    return render_template(
        'dashboard.html',
        user_name=session.get("user", "User"),
        active_page="chat"
    )

@app.route('/video-call', methods=['GET'])
def video_call():
    if "user_id" not in session:
        flash("Please login to continue.", "error")
        return redirect(url_for("login"))
    print(f"[route] /video-call loaded for user_id={session.get('user_id')}")
    return render_template(
        "dashboard.html",
        user_name=session.get("user", "User"),
        active_page="video"
    )

@app.route('/suicide_score')
def suicide_score():
    user_id = session.get("user_id")
    if not user_id:
        flash("Please login to view your mental score.", "error")
        return redirect(url_for("login"))

    result = analyze_suicide_and_notify(user_id)
    if result["suicide_percentage"] is None:
        return jsonify({"error": f"Unable to evaluate suicide score for user_id={user_id}"}), 500

    return jsonify({
        "suicide_percentage": result["suicide_percentage"],
        "action_taken": result["action_taken"]
    })

    
    
@app.route('/mental_score')
def mental_score():
    user_id = session.get("user_id")
    if not user_id:
        flash("Please login to view your mental score.", "error")
        return redirect(url_for("login"))

    chat_file = get_chat_file(user_id)
    embedding_path = './model/embeddings.npy'

    if not os.path.exists(chat_file):
        flash("No chat history found to calculate score.", "warning")
        return jsonify({"error": f"File not found for userid: {user_id} and filepath: {chat_file}"}), 500

    try:
        if RAGSimilarityClassifier is None:
            return jsonify({"error": "RAG classifier unavailable in this runtime."}), 500
        dataset_path = get_dataset_path()
        if not dataset_path:
            return jsonify({"error": "Dataset unavailable from Google Drive and no local fallback found."}), 500
        classifier = RAGSimilarityClassifier(dataset_path, embedding_path, filepath=chat_file)
        predicted_labels, label_counts = classifier.predict_labels()

        # Calculate an overall score (example: stress = 1, anxiety = 2, depression = 3, PTSD = 4)
        mood_weights = {
            "normal": 4,
            "stress": 3,
            "anxiety": 2,
            "depression": 1,
            "PTSD": 0
        }

        total_score = sum(mood_weights.get(label, 0) * count for label, count in label_counts.items())
        total_msgs = sum(label_counts.values())
        mood_score = round(total_score / total_msgs, 2) if total_msgs > 0 else 0

        # Save to DB
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO mental_scores (userid, score, label_counts)
                VALUES (?, ?, ?)
            ''', (user_id, mood_score, json.dumps(label_counts)))
            conn.commit()

        return jsonify({"mood_score": mood_score})

    except Exception as e:
        return jsonify({"error": f"Error calculating score: {str(e)}"}), 500


@app.route('/get_recommendation')
def get_recommendation():
    user_id = session.get("user_id")

    if not user_id:
        flash("Please login first to see recommendations.", "error")
        return redirect(url_for("login"))

    chat_file = get_chat_file(user_id)
    rec_file = get_recommendation_file(user_id)

    print(f"[get_recommendation] user_id={user_id}, chat_file={chat_file}, rec_file={rec_file}")

    # If recommendation doesn't exist, generate it
    if not os.path.exists(rec_file):
        if os.path.exists(chat_file):
            print(f"[get_recommendation] recommendation missing, generating for user_id={user_id}")
            get_counselor_ai().generate_recommendation(chat_file, user_id)
        else:
            print(f"[get_recommendation] chat history missing for user_id={user_id}")
            return ("No chat history found to generate recommendation.", 404)

    # Read the recommendation
    with open(rec_file, "r", encoding="utf-8") as file:
        recommendation_text = file.read()

        return recommendation_text

@app.route('/recommendation')
def recommendation():
    if "user_id" not in session:
        flash("Please login to continue.", "error")
        return redirect(url_for("login"))
    print(f"[route] /recommendation loaded for user_id={session.get('user_id')}")
    return render_template(
        "dashboard.html",
        user_name=session.get("user", "User"),
        active_page="recommendation"
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['userid']
        password = request.form['password']

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE (userid=? OR email=?) AND password=?', 
                           (username_or_email, username_or_email, password))
            user = cursor.fetchone()
        
        if user:
            session['user'] = user[1]  # fullname
            session['user_id'] = user[6]  # userid
            session['user_mail'] = user[4]
            session['session_id'] = str(uuid.uuid4())  # generate new unique session_id
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please sign up if you don't have an account.", "error")
            return redirect(url_for('signup'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        age_raw = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        email = request.form.get('email', '').strip().lower()
        mobile = request.form.get('mobile', '').strip()
        userid = request.form.get('userid', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm-password', '')

        # Fallback: build fullname from first/last if hidden fullname is missing.
        if not fullname:
            fullname = f"{first_name} {last_name}".strip()

        # Fallback: derive userid from email if missing.
        if not userid and email:
            userid = email.split("@")[0]

        if not all([fullname, age_raw, gender, email, mobile, userid, password, confirm_password]):
            flash("Please fill all required signup fields.", "error")
            return redirect(url_for('signup'))

        try:
            age = int(age_raw)
        except ValueError:
            flash("Age must be a valid number.", "error")
            return redirect(url_for('signup'))

        if age <= 0:
            flash("Age must be greater than zero.", "error")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('signup'))

        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (fullname, age, gender, email, mobile, userid, password)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (fullname, age, gender, email, mobile, userid, password))
                conn.commit()
                flash("Signup successful. Please login!", "success")
                return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("User with this email or username already exists!", "error")
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json(silent=True) or {}

    if 'user_input' not in data:
        return jsonify({"error": "Missing 'user_input' in request"}), 400

    user_input = str(data.get('user_input', '')).strip()
    if not user_input:
        return jsonify({"error": "Empty 'user_input' is not allowed"}), 400

    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"error": "No active session."}), 403

    try:
        print(f"[get_response] request user_id={user_id}, input_len={len(user_input)}")
        ai_response = get_chatbot().chat(user_id, user_input)
        chat_file = get_chat_file(user_id)
        print(f"[get_response] response generated user_id={user_id}, chat_file_exists={os.path.exists(chat_file)}")
        if not ai_response:
            return jsonify({"response": "I am here with you. Could you share a little more?"})
        return jsonify({"response": str(ai_response)})
    except ValueError as e:
        # Configuration/runtime validation issues (e.g., missing API key)
        print(f"[get_response] ValueError for user_id={user_id}: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"[get_response] Unexpected error for user_id={user_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": "I am unable to answer right now. Please try again in a moment."}), 500

@app.route("/end_chat", methods=["POST"])
def end_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "No active session found."}), 403

    try:
        print(f"[end_chat] starting for user_id={user_id}")
        # Save latest file-based history snapshot.
        bot = get_chatbot()
        previous_chat_history = bot.load_chat_history(user_id)
        bot.save_chat_history(user_id, previous_chat_history)
        bot.clear_memory(user_id)

        # Run recommendation pipeline immediately after ending chat.
        chat_file = get_chat_file(user_id)
        if os.path.exists(chat_file):
            print(f"[end_chat] chat history found for user_id={user_id}. Generating recommendation.")
            get_counselor_ai().generate_recommendation(chat_file, user_id)
        else:
            print(f"[end_chat] Chat history file not found for user_id={user_id}: {chat_file}")

        # Analyze saved chat with suicide detector and send email if triggered.
        analyze_suicide_and_notify(user_id)

        return jsonify({
            "message": "Chat ended and recommendation generated.",
            "redirect_url": url_for("recommendation")
        })
    except Exception as e:
        print(f"[end_chat] Unexpected error for user_id={user_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Unable to end chat at the moment."}), 500

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.clear()  # Clear all session data including session_id
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('login'))

try:
    init_db()
except Exception as db_init_error:
    print(f"[startup] init_db failed: {db_init_error}")

if __name__ == '__main__':
    app.run(debug=True)
