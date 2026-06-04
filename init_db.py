import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Force load environment variables from .env
load_dotenv(override=True)

def init_postgres():
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "hemoscan")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pwd = os.environ.get("POSTGRES_PASSWORD", "Sweethome123")

    print(f"Connecting to PostgreSQL server at {host}:{port} as user '{user}'...")
    
    # Connect to the default 'postgres' database to check/create the target database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=pwd
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to default 'postgres' database: {e}")
        print("Please make sure PostgreSQL is running and credentials are correct.")
        return False

    # Check if the database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cursor.fetchone()
    
    if not exists:
        print(f"Database '{db_name}' does not exist. Creating...")
        try:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully.")
        except Exception as e:
            print(f"Error creating database '{db_name}': {e}")
            cursor.close()
            conn.close()
            return False
    else:
        print(f"Database '{db_name}' already exists.")
        
    cursor.close()
    conn.close()

    # Now, connect to the target database and execute the schema
    print(f"Connecting to database '{db_name}' to run schema...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db_name,
            user=user,
            password=pwd
        )
        cursor = conn.cursor()
        
        # Read and execute schema
        schema_path = "postgres_schema.sql"
        if not os.path.exists(schema_path):
            print(f"Error: Schema file '{schema_path}' not found.")
            return False
            
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
            
        cursor.execute(schema_sql)
        conn.commit()
        print("Database schema initialized successfully (tables created and admin user seeded).")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error initializing schema in database '{db_name}': {e}")
        return False

if __name__ == "__main__":
    success = init_postgres()
    if success:
        print("Database setup completed successfully!")
    else:
        print("Database setup failed.")
