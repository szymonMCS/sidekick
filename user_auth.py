import sqlite3
import bcrypt
from typing import Optional, Tuple
import os

class UserAuth:
    """Handle user registration, login, and session managment"""

    def __init__(self, db_path: str = "sandbox/users.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "sandbox", exist_ok = True)
        self.init_db()

    def init_db(self):
        """Initialize users database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Register a new user
        Returns:
            (success: bool, message: str)
        """
        #Validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"

        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"

        try:
            #hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'),
            bcrypt.gensalt())

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                (username, password_hash)
            )

            conn.commit()
            conn.close()

            return True, f"User '{username}' registered succesfully!"

        except sqlite3.IntegrityError:
            return False, f"Username '{username}' alread exists"
        except Exception as e:
            return False, f"Registration error: {str(e)}"

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate user
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id, password_hash FROM users WHERE username = ?', (username,)
            )

            result = cursor.fetchone()

            if not result:
                conn.close()
                return False, "Invalid username or password"

            user_id, stored_hash = result

            #verufy password
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                cursor.execute(
                    'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,)
                )
                conn.commit()
                conn.close()

                return True, f"Welcome back, {username}!"
            else:
                conn.close()
                return False, "Invalid username or password"

        except Exception as e:
            return False, f"Login error: {str(e)}"

    def get_user_stats(self, username: str) -> Optional[dict]:
        """Get user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id, created_at, last_login FROM users WHERE username = ?',
                (username,)
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'user_id': result[0],
                    'created_at': result[1],
                    'last_login': result[2]
                }
            return None

        except Exception:
            return None