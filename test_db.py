import os
import psycopg2

def get_connection():
    dsn = os.environ.get('postgres://discord_bot_data_user:BdimL061db6iEusp0P8ftr4OnyyLj2Th@dpg-cnoph1vsc6pc73b7d3ug-a.oregon-postgres.render.com/discord_bot_data')
    return psycopg2.connect("postgres://discord_bot_data_user:BdimL061db6iEusp0P8ftr4OnyyLj2Th@dpg-cnoph1vsc6pc73b7d3ug-a.oregon-postgres.render.com/discord_bot_data")


with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('CREATE DATABASE USERLEVEL')
