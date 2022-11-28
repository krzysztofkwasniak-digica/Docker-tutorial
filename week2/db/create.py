import argparse
import sqlite3
from pathlib import Path
from sqlite3 import Error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=Path, default="db/store.db" ,help="The destination of created DB")
    return parser.parse_args()

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    args = parse_args()
    create_connection(args.dest)