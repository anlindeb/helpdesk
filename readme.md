sudo apt update
sudo apt install -y python3 python3-venv python3-pip
sudo apt install sqlite3

source venv/bin/activate

python init_helpdesk_db.py
python init_users_db.py

pip install openai faiss-cpu numpy

"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"

python build_faiss_index.py

pip install fastapi uvicorn openai faiss-cpu numpy

mkdir static


uvicorn main:app --host 0.0.0.0 --port 8000 --reload


merge tickets - 