#!/usr/bin/env python3
import sqlite3
import random

DB_FILENAME = "helpdesk.db"

conn = sqlite3.connect(DB_FILENAME)
cursor = conn.cursor()

# Fetch all user IDs
cursor.execute("SELECT user_id FROM users;")
user_ids = [row[0] for row in cursor.fetchall()]
if not user_ids:
    print("No users found in the database; cannot backfill.")
    conn.close()
    exit(1)

# Fetch all ticket IDs
cursor.execute("SELECT ticket_id FROM tickets;")
tickets = [row[0] for row in cursor.fetchall()]

# Assign a random user to each ticket
for ticket_id in tickets:
    assigned_user = random.choice(user_ids)
    cursor.execute(
        "UPDATE tickets SET requester_id = ? WHERE ticket_id = ?;",
        (assigned_user, ticket_id)
    )

conn.commit()
print(f"Backfilled {len(tickets)} tickets with random requester_ids.")
conn.close()