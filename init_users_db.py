#!/usr/bin/env python3
import sqlite3
import datetime
import random

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Path to your existing SQLite database
DB_FILENAME = "helpdesk.db"

# -------------------------------------------------------------
# STEP A: Define a small pool of synthetic names and departments
# -------------------------------------------------------------
first_names = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kara", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn",
    "Rachel", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander", "Yara", "Zack"
]
last_names = [
    "Anderson", "Brown", "Chen", "Davis", "Evans", "Garcia", "Hernandez",
    "Johnson", "Kim", "Lee", "Martinez", "Nguyen", "O'Neil", "Patel",
    "Robinson", "Smith", "Taylor", "Upton", "Valdez", "Walker", "Xu", "Young", "Zhang"
]
departments = ["IT", "HR", "Finance", "Marketing", "Sales", "Support", "R&D"]

# How many synthetic users to create
NUM_SYNTHETIC_USERS = 50


# -------------------------------------------------------------
# STEP B: Connect to helpdesk.db (it must already exist)
# -------------------------------------------------------------
conn = sqlite3.connect(DB_FILENAME)
cursor = conn.cursor()

# -------------------------------------------------------------
# STEP C: Create the 'users' table if it doesn’t exist
# -------------------------------------------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id     INTEGER   PRIMARY KEY AUTOINCREMENT,
        full_name   TEXT      NOT NULL,
        email       TEXT      NOT NULL UNIQUE,
        department  TEXT      NOT NULL,
        created_at  TEXT      NOT NULL
    );
""")
conn.commit()


# -------------------------------------------------------------
# STEP D: Generate and insert synthetic user rows
# -------------------------------------------------------------
def random_name():
    """Pick a random first+last name and return as a single string."""
    fn = random.choice(first_names)
    ln = random.choice(last_names)
    return f"{fn} {ln}"

def make_email(full_name):
    """
    Derive a synthetic email from full_name, e.g. "Alice Johnson" → "alice.johnson@example.com"
    If collision occurs, append a random number.
    """
    base = full_name.lower().replace(" ", ".")
    return f"{base}@example.com"

insert_sql = """
    INSERT OR IGNORE INTO users (full_name, email, department, created_at)
    VALUES (?, ?, ?, ?);
"""

count_inserted = 0
existing_emails = set()

for _ in range(NUM_SYNTHETIC_USERS):
    # 1) Generate a random name
    name = random_name()

    # 2) Generate a base email
    email = make_email(name)

    # 3) If this email is already used, append a 2-digit random number
    if email in existing_emails:
        number = random.randint(10, 99)
        email = f"{email.split('@')[0]}{number}@example.com"

    existing_emails.add(email)

    # 4) Pick a random department
    dept = random.choice(departments)

    # 5) created_at = today (ISO format)
    created_at = datetime.date.today().isoformat()

    # 6) Insert into the table
    try:
        cursor.execute(insert_sql, (name, email, dept, created_at))
        if cursor.rowcount == 1:
            count_inserted += 1
    except sqlite3.IntegrityError as e:
        print(f"Skipping duplicate email insertion: {email} ({e})")

conn.commit()
print(f"Inserted {count_inserted} synthetic users into 'users' table.")
conn.close()
