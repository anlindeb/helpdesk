#!/usr/bin/env python3
import sqlite3
import json
import os
import sys

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# 1) Path to the SQLite database file we want to create/use.
DB_FILENAME = "helpdesk.db"

# 2) Path to your JSON file with all tickets.
JSON_FILENAME = "tickets.json"


# -------------------------------------------------------------
# STEP A: Verify that tickets.json exists
# -------------------------------------------------------------
if not os.path.isfile(JSON_FILENAME):
    print(f"Error: Cannot find '{JSON_FILENAME}'. Make sure it is in this folder.")
    sys.exit(1)

# -------------------------------------------------------------
# STEP B: Load JSON data into memory
# -------------------------------------------------------------
with open(JSON_FILENAME, "r", encoding="utf-8") as f:
    try:
        tickets = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        sys.exit(1)

# Expecting tickets to be a list of dicts
if not isinstance(tickets, list):
    print("Error: JSON root is not a list. It should be an array of ticket objects.")
    sys.exit(1)

# -------------------------------------------------------------
# STEP C: Connect (or create) the SQLite database
# -------------------------------------------------------------
conn = sqlite3.connect(DB_FILENAME)
cursor = conn.cursor()

# -------------------------------------------------------------
# STEP D: Create the tickets table if it doesn’t already exist
# -------------------------------------------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id    INTEGER   PRIMARY KEY,
        issue        TEXT      NOT NULL,
        status       TEXT      NOT NULL,
        resolution   TEXT,
        date_opened  TEXT      NOT NULL,
        date_closed  TEXT
    );
""")
conn.commit()

# -------------------------------------------------------------
# STEP E: Insert JSON rows into the tickets table
# -------------------------------------------------------------
# We’ll do an UPSERT-like approach: if a ticket_id already exists, skip it.
insert_sql = """
    INSERT OR IGNORE INTO tickets
      (ticket_id, issue, status, resolution, date_opened, date_closed)
    VALUES (?, ?, ?, ?, ?, ?);
"""

count_inserted = 0
for entry in tickets:
    # Validate required fields
    required_fields = ["TicketID", "Issue", "Status", "DateOpened"]
    missing = [f for f in required_fields if f not in entry]
    if missing:
        print(f"Skipping entry because missing fields {missing}: {entry}")
        continue

    t_id = entry["TicketID"]
    issue = entry["Issue"]
    status = entry["Status"]
    resolution = entry.get("Resolution")  # may be None
    date_opened = entry["DateOpened"]
    # date_closed might be null or missing; store as None in that case
    date_closed = entry.get("DateClosed")

    try:
        cursor.execute(
            insert_sql,
            (t_id, issue, status, resolution, date_opened, date_closed),
        )
        count_inserted += cursor.rowcount
    except sqlite3.IntegrityError as e:
        print(f"Failed to insert TicketID={t_id}: {e}")

conn.commit()
print(f"Inserted (or ignored) {count_inserted} rows into '{DB_FILENAME}'.")
conn.close()