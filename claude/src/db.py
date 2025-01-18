import sqlite3

# Initialize the database and create the table if it doesn't exist
def init_db():
    conn = sqlite3.connect("semantics.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS t_semantics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            method TEXT,
            input_text TEXT,
            chart_path TEXT
        )
    """)
    conn.commit()
    return conn

# Save results to the database
def save_result(conn, model_name, method, input_text, chart_path):
    c = conn.cursor()
    c.execute("""
        INSERT INTO t_semantics (model_name, method, input_text, chart_path)
        VALUES (?, ?, ?, ?)
    """, (model_name, method, input_text, chart_path))
    conn.commit()

# Fetch all results from the database
def fetch_results():
    conn = sqlite3.connect("semantics.db")
    df = pd.read_sql("SELECT * FROM t_semantics", conn)
    conn.close()
    return df