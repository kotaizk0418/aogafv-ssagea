import os
import psycopg2

def get_connection():
    return

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('CREATE DATABASE USERLEVEL')
