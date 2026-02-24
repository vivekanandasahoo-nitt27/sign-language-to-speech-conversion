import sqlite3
from pathlib import Path

# ⭐ DB file inside v2 folder
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app_v2.db"


# ================= DB CONNECTION =================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ================= INIT DB =================
def init_db():
    db = get_db()
    cur = db.cursor()

    # ---------- USERS ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---------- SENTENCES MEMORY ⭐ ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentences(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        sentence TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    db.commit()
    db.close()


# ================= HELPERS =================

def create_user(username: str, password: str):
    db = get_db()
    cur = db.cursor()

    try:
        cur.execute(
            "INSERT INTO users(username, password) VALUES (?, ?)",
            (username, password)
        )
        db.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        user_id = None

    db.close()
    return user_id


def get_user(username: str, password: str):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT id, username FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cur.fetchone()

    db.close()
    return user


def insert_sentence(user_id: int, sentence: str):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "INSERT INTO sentences(user_id, sentence) VALUES (?, ?)",
        (user_id, sentence)
    )

    db.commit()
    db.close()


def get_last_sentences(user_id: int, limit: int = 3):
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        SELECT sentence FROM sentences
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))

    rows = cur.fetchall()
    db.close()

    # return oldest → newest ⭐ important for context order
    return [r["sentence"] for r in rows[::-1]]