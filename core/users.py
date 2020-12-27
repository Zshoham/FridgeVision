import os
import sqlite3
import json


class Users:
    """
    This class represents the users in the database.
    Two main methods are exposed, load_or_create is used for reading the user data.
    and set_user_groceries updates the groceries the user expects to need.
    """
    def __init__(self, db_path):
        db_connection = sqlite3.connect(os.path.join(db_path, "users.db"))
        self.db = db_connection.cursor()
        self.db.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, groceries TEXT)")
        db_connection.commit()

    def load_or_create(self, name):
        self.db.execute("SELECT * FROM users WHERE username = ?", (name,))
        users = self.db.fetchall()
        self.db.connection.commit()

        if not users:
            self.db.execute("INSERT INTO users (username, groceries) VALUES (?,?)", (name, ""))
            self.db.connection.commit()
            return None
        else:
            return json.loads(users[0][1])

    def set_user_groceries(self, name, groceries):
        json_groceries = json.dumps(groceries)
        self.db.execute("UPDATE users SET groceries = ? WHERE username = ?", (json_groceries, name))
        self.db.connection.commit()

    def close(self):
        self.db.connection.close()
