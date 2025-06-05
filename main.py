#!/usr/bin/env python3
import os
import sqlite3
import pickle
import numpy as np
import faiss
from datetime import date
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DB_FILENAME        = "helpdesk.db"         # SQLite file from Step 1
FAISS_INDEX_FILE   = "faiss_index.bin"     # FAISS index from Step 2
MAPPING_FILE       = "ticket_mapping.pkl"  # ticket_id ↔ text mapping

EMBEDDING_MODEL    = "text-embedding-3-small"   # 1536‐dim embeddings
CHAT_MODEL         = "gpt-4o-mini"              # chat-capable model
TOP_K              = 5                           # how many neighbors to retrieve

# -------------------------------------------------------------
# INITIALIZATION (load FAISS index + mapping, OpenAI client, user lookup)
# -------------------------------------------------------------
# Ensure required files exist
if not os.path.isfile(DB_FILENAME):
    raise RuntimeError(f"Database file `{DB_FILENAME}` not found. Run Step 1 first.")
if not os.path.isfile(FAISS_INDEX_FILE):
    raise RuntimeError(f"FAISS index `{FAISS_INDEX_FILE}` not found. Run Step 2 first.")
if not os.path.isfile(MAPPING_FILE):
    raise RuntimeError(f"Mapping file `{MAPPING_FILE}` not found. Run Step 2 first.")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)

# Load ticket mapping (ticket_id ↔ combined text)
with open(MAPPING_FILE, "rb") as f:
    ticket_mapping: List[dict] = pickle.load(f)

# Extract parallel lists for quick lookup
ticket_ids   = [entry["ticket_id"] for entry in ticket_mapping]
ticket_texts = [entry["text"] for entry in ticket_mapping]

# Build a user_lookup: user_id → full_name
user_lookup = {}
conn_tmp = sqlite3.connect(DB_FILENAME)
cur_tmp = conn_tmp.cursor()
cur_tmp.execute("SELECT user_id, full_name FROM users;")
for uid, name in cur_tmp.fetchall():
    user_lookup[uid] = name
conn_tmp.close()

# Initialize OpenAI client
client = OpenAI()

# -------------------------------------------------------------
# FASTAPI APP + CORS
# -------------------------------------------------------------
app = FastAPI(
    title="IT Helpdesk RAG Chatbot",
    description="A FastAPI service that serves a helpdesk ticket DB + RAG-powered Chat endpoint.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Pydantic models for Tickets / Users / ChatRequest
# -------------------------------------------------------------
class TicketBase(BaseModel):
    issue:        str         = Field(..., example="User cannot connect to WiFi.")
    status:       str         = Field(..., example="Open")
    resolution:   Optional[str] = Field(None, example="Reset WiFi adapter and reboot machine.")
    date_opened:  date        = Field(..., example="2025-06-03")
    date_closed:  Optional[date] = Field(None, example="2025-06-05")
    requester_id: Optional[int]  = Field(None, example=3)  # references users.user_id

class TicketCreate(TicketBase):
    pass

class TicketUpdate(BaseModel):
    issue:        Optional[str] = None
    status:       Optional[str] = None
    resolution:   Optional[str] = None
    date_opened:  Optional[date] = None
    date_closed:  Optional[date] = None
    requester_id: Optional[int]  = None

class TicketInDB(TicketBase):
    ticket_id: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    full_name:  str = Field(..., example="Alice Johnson")
    email:      str = Field(..., example="alice.johnson@example.com")
    department: str = Field(..., example="IT")

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    full_name:  Optional[str] = None
    email:      Optional[str] = None
    department: Optional[str] = None

class UserInDB(UserBase):
    user_id:    int
    created_at: date

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    question: str = Field(..., example="My printer is showing a paper jam.")

# -------------------------------------------------------------
# UTILITY: get a SQLite connection (per-request)
# -------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# -------------------------------------------------------------
# /tickets endpoints (CRUD)
# -------------------------------------------------------------
@app.get("/tickets/", response_model=List[TicketInDB])
def read_all_tickets(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tickets ORDER BY ticket_id ASC;")
    rows = cursor.fetchall()
    return [TicketInDB(**dict(row)) for row in rows]

@app.get("/tickets/{ticket_id}", response_model=TicketInDB)
def read_ticket(ticket_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketInDB(**dict(row))

@app.post("/tickets/", response_model=TicketInDB, status_code=201)
def create_ticket(payload: TicketCreate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO tickets(issue, status, resolution, date_opened, date_closed, requester_id)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            payload.issue,
            payload.status,
            payload.resolution,
            payload.date_opened.isoformat(),
            payload.date_closed.isoformat() if payload.date_closed else None,
            payload.requester_id
        )
    )
    db.commit()
    new_id = cursor.lastrowid
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?;", (new_id,))
    row = cursor.fetchone()
    return TicketInDB(**dict(row))

@app.put("/tickets/{ticket_id}", response_model=TicketInDB)
def update_ticket(ticket_id: int, payload: TicketUpdate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    fields = []
    values = []
    for field_name, field_value in payload.dict(exclude_unset=True).items():
        if field_name in {"date_opened", "date_closed"} and field_value is not None:
            fields.append(f"{field_name} = ?")
            values.append(field_value.isoformat())
        else:
            fields.append(f"{field_name} = ?")
            values.append(field_value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    values.append(ticket_id)
    set_clause = ", ".join(fields)
    sql = f"UPDATE tickets SET {set_clause} WHERE ticket_id = ?;"
    cursor.execute(sql, tuple(values))
    db.commit()

    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    updated_row = cursor.fetchone()
    return TicketInDB(**dict(updated_row))

# -------------------------------------------------------------
# /users endpoints (CRUD)
# -------------------------------------------------------------
@app.get("/users/", response_model=List[UserInDB])
def read_all_users(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users ORDER BY user_id ASC;")
    rows = cursor.fetchall()
    return [UserInDB(**dict(row)) for row in rows]

@app.get("/users/{user_id}", response_model=UserInDB)
def read_user(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInDB(**dict(row))

@app.post("/users/", response_model=UserInDB, status_code=201)
def create_user(payload: UserCreate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    created_at = date.today().isoformat()
    cursor.execute(
        """
        INSERT INTO users(full_name, email, department, created_at)
        VALUES (?, ?, ?, ?);
        """,
        (payload.full_name, payload.email, payload.department, created_at)
    )
    db.commit()
    new_id = cursor.lastrowid
    cursor.execute("SELECT * FROM users WHERE user_id = ?;", (new_id,))
    row = cursor.fetchone()
    # Update user_lookup so new user appears in future /chat lookups
    user_lookup[new_id] = payload.full_name
    return UserInDB(**dict(row))

@app.put("/users/{user_id}", response_model=UserInDB)
def update_user(user_id: int, payload: UserUpdate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    fields = []
    values = []
    for field_name, field_value in payload.dict(exclude_unset=True).items():
        fields.append(f"{field_name} = ?")
        values.append(field_value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    values.append(user_id)
    set_clause = ", ".join(fields)
    sql = f"UPDATE users SET {set_clause} WHERE user_id = ?;"
    cursor.execute(sql, tuple(values))
    db.commit()

    # If full_name changed, update user_lookup
    if "full_name" in payload.dict(exclude_unset=True):
        user_lookup[user_id] = payload.full_name  # update name in lookup

    cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
    updated_row = cursor.fetchone()
    return UserInDB(**dict(updated_row))

# -------------------------------------------------------------
# /chat endpoint (RAG-powered helpdesk) with requester info
# -------------------------------------------------------------
@app.post("/chat")
def chat(request: ChatRequest, db: sqlite3.Connection = Depends(get_db)):
    user_q = request.question.strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1) Embed the user’s question
    try:
        resp = client.embeddings.create(
            input=[user_q],
            model=EMBEDDING_MODEL
        )
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Embedding error: {e}")

    q_vec = np.array([resp.data[0].embedding], dtype="float32")

    # 2) Search FAISS for top-K nearest neighbors
    D, I = index.search(q_vec, TOP_K)
    neighbor_indices = I[0].tolist()

    # 3) Retrieve and label ticket texts with requester name
    retrieved_tickets = []
    cursor = db.cursor()
    for i, idx in enumerate(neighbor_indices):
        if 0 <= idx < len(ticket_ids):
            tid = ticket_ids[idx]
            blob = ticket_texts[idx]
            # Split into issue & resolution
            if "\nResolution: " in blob:
                issue_part, resolution_part = blob.split("\nResolution: ", 1)
            else:
                issue_part = blob
                resolution_part = ""
            # Fetch requester_id from tickets table
            cursor.execute("SELECT requester_id FROM tickets WHERE ticket_id = ?;", (tid,))
            row = cursor.fetchone()
            requester_name = None
            if row and row["requester_id"] and row["requester_id"] in user_lookup:
                requester_name = user_lookup[row["requester_id"]]
            # Build snippet header
            if requester_name:
                snippet_header = f"[Ticket {tid} opened by {requester_name}]"
            else:
                snippet_header = f"[Ticket {tid}]"
            retrieved_tickets.append({
                "ticket_id": tid,
                "issue": issue_part,
                "resolution": resolution_part,
                "requester": requester_name,
                "distance": float(D[0][i])
            })

    # 4) Build structured prompt for ChatCompletion
    system_msg = (
        "You are an expert IT helpdesk assistant. Use the following ticket information to "
        "answer the user’s question. Each snippet is labeled with its Ticket ID, Requester, Issue, and Resolution."
    )

    prompt = ""
    for i, rt in enumerate(retrieved_tickets, start=1):
        prompt += (
            f"Snippet {i}:\n"
            f"{snippet_header}\n"
            f"Issue: {rt['issue']}\n"
            f"Resolution: {rt['resolution']}\n\n"
        )
    prompt += f"User question: {user_q}\n\nProvide a clear, concise resolution as an IT support agent."

    # 5) Call ChatCompletion
    try:
        chat_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2
        )
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI ChatCompletion error: {e}")

    answer = chat_resp.choices[0].message.content

    return {
        "answer": answer,
        "retrieved": retrieved_tickets
    }

# -------------------------------------------------------------
# Serve static files (if `static/` exists)
# -------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")