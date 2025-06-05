#!/usr/bin/env python3
import sqlite3
import pickle
import os
import time

import numpy as np
import faiss
from openai import OpenAI, OpenAIError

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DB_FILENAME       = "helpdesk.db"         # Your SQLite file from Step 1
FAISS_INDEX_FILE  = "faiss_index.bin"     # Where we’ll save the FAISS index
MAPPING_FILE      = "ticket_mapping.pkl"  # Where we’ll save (ticket_id → text) mapping

EMBEDDING_MODEL   = "text-embedding-3-small"   # Embedding model (1536-dim)
BATCH_SIZE        = 20                    # How many tickets to embed per API call
EMBEDDING_DIM     = 1536                  # Dimension of the chosen model’s output
RATE_LIMIT_DELAY  = 1.0                   # Seconds to wait between embedding batches


# -------------------------------------------------------------
# STEP A: Connect to SQLite and fetch all tickets
# -------------------------------------------------------------
if not os.path.isfile(DB_FILENAME):
    print(f"Error: `{DB_FILENAME}` not found. Run Step 1 first to create the database.")
    exit(1)

conn = sqlite3.connect(DB_FILENAME)
cursor = conn.cursor()

cursor.execute("""
    SELECT ticket_id, issue, IFNULL(resolution, "")
      FROM tickets
    ORDER BY ticket_id ASC;
""")
rows = cursor.fetchall()
conn.close()

if not rows:
    print("Error: No tickets found in the database.")
    exit(1)

print(f"Fetched {len(rows)} tickets from `{DB_FILENAME}`.")


# -------------------------------------------------------------
# STEP B: Build text blobs and prepare mapping
# -------------------------------------------------------------
ticket_mapping = []
for ticket_id, issue, resolution in rows:
    combined = issue.strip()
    if resolution.strip():
        combined += "\nResolution: " + resolution.strip()
    ticket_mapping.append({"ticket_id": ticket_id, "text": combined})

all_texts = [entry["text"] for entry in ticket_mapping]


# -------------------------------------------------------------
# STEP C: Initialize OpenAI client & FAISS index
# -------------------------------------------------------------
client = OpenAI()  # Make sure OPENAI_API_KEY is set in your env
index = faiss.IndexFlatL2(EMBEDDING_DIM)
print(f"Created new FAISS IndexFlatL2 with dimension {EMBEDDING_DIM}.")


# -------------------------------------------------------------
# STEP D: Embed in batches and add to FAISS
# -------------------------------------------------------------
def chunk_list(lst, size):
    """Yield successive chunks of `size` from list `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

num_tickets = len(all_texts)
print(f"Beginning embedding of {num_tickets} texts in batches of {BATCH_SIZE}...")

start_time = time.time()
vector_id = 0

for batch_texts in chunk_list(all_texts, BATCH_SIZE):
    try:
        resp = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_MODEL
        )
    except OpenAIError as e:
        print("OpenAI API error:", e)
        exit(1)

    # Now use resp.data instead of resp["data"]
    embedding_vectors = []
    for item in resp.data:
        # Each `item` has an `.embedding` attribute (a list of floats)
        embedding_vectors.append(item.embedding)

    # Convert to a (N × EMBEDDING_DIM) float32 NumPy array
    np_vectors = np.array(embedding_vectors, dtype="float32")

    # Add to FAISS index
    index.add(np_vectors)

    vector_id += np_vectors.shape[0]
    elapsed = time.time() - start_time
    print(f"  → Added batch of {len(batch_texts)} vectors → total so far: {vector_id}  (elapsed {elapsed:.1f}s)")
    time.sleep(RATE_LIMIT_DELAY)

assert vector_id == num_tickets, "FAISS index size mismatch! Check embedding loop."


# -------------------------------------------------------------
# STEP E: Save FAISS index and mapping to disk
# -------------------------------------------------------------
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved to `{FAISS_INDEX_FILE}`.")

with open(MAPPING_FILE, "wb") as f:
    pickle.dump(ticket_mapping, f)
print(f"Ticket mapping (ticket_id ↔ text) saved to `{MAPPING_FILE}`.")

total_time = time.time() - start_time
print(f"Completed embedding & indexing {num_tickets} tickets in {total_time:.1f}s.")