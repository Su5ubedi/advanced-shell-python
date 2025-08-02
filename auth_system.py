#!/usr/bin/env python3
"""
auth_system.py - User authentication and authorization system
"""

import hashlib
import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    STANDARD = "standard"
    GUEST = "guest"


class Permission(Enum):
    """File permission enumeration"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


@dataclass
class User:
    """Represents a user in the system"""
    username: str
    password_hash: str
    role: UserRole
    home_directory: str
    permissions: Dict[str, List[Permission]] = None  # file_path -> permissions

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {}


class AuthenticationSystem:
    """Handles user authentication and authorization"""

    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.current_user: Optional[User] = None
        self.users: Dict[str, User] = {}
        self.load_users()

    def load_users(self) -> None:
        """Load users from file or create default users"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data['users']:
                        user = User(
                            username=user_data['username'],
                            password_hash=user_data['password_hash'],
                            role=UserRole(user_data['role']),
                            home_directory=user_data['home_directory'],
                            permissions=user_data.get('permissions', {})
                        )
                        self.users[user.username] = user
            except Exception as e:
                print(f"Warning: Could not load users file: {e}")
                self.create_default_users()
        else:
            self.create_default_users()

    def create_default_users(self) -> None:
        """Create default users for the system with restricted write permissions"""
        # Create admin user with full access
        admin_hash = self._hash_password("admin123")
        admin_user = User(
            username="admin",
            password_hash=admin_hash,
            role=UserRole.ADMIN,
            home_directory="/home/admin",
            permissions={
                "/": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/etc": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/var": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/usr": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/bin": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/sbin": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE]
            }
        )

        # Create standard user with read-only access to most directories
        user_hash = self._hash_password("user123")
        standard_user = User(
            username="user",
            password_hash=user_hash,
            role=UserRole.STANDARD,
            home_directory="/home/user",
            permissions={
                "/home/user": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ, Permission.EXECUTE],
                "/usr": [Permission.READ, Permission.EXECUTE],
                "/bin": [Permission.READ, Permission.EXECUTE],
                "/var": [Permission.READ],
                "/etc": [Permission.READ]
            }
        )

        # Create guest user with very limited read-only access
        guest_hash = self._hash_password("guest123")
        guest_user = User(
            username="guest",
            password_hash=guest_hash,
            role=UserRole.GUEST,
            home_directory="/home/guest",
            permissions={
                "/home/guest": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ],
                "/usr": [Permission.READ],
                "/bin": [Permission.READ]
            }
        )

        self.users = {
            "admin": admin_user,
            "user": standard_user,
            "guest": guest_user
        }

        self.save_users()

    def save_users(self) -> None:
        """Save users to file"""
        try:
            data = {
                'users': [
                    {
                        'username': user.username,
                        'password_hash': user.password_hash,
                        'role': user.role.value,
                        'home_directory': user.home_directory,
                        'permissions': user.permissions
                    }
                    for user in self.users.values()
                ]
            }
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save users file: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password"""
        if username not in self.users:
            return False

        user = self.users[username]
        password_hash = self._hash_password(password)
        
        if user.password_hash == password_hash:
            self.current_user = user
            return True
        return False

    def logout(self) -> None:
        """Logout current user"""
        self.current_user = None

    def is_authenticated(self) -> bool:
        """Check if a user is currently authenticated"""
        return self.current_user is not None

    def get_current_user(self) -> Optional[User]:
        """Get the currently authenticated user"""
        return self.current_user

    def has_permission(self, file_path: str, permission: Permission) -> bool:
        """Check if current user has permission for a file with enhanced logic"""
        if not self.current_user:
            return False

        # Admin has all permissions
        if self.current_user.role == UserRole.ADMIN:
            return True

        # Normalize file path
        file_path = os.path.abspath(file_path)
        
        # Check user-specific permissions first
        if file_path in self.current_user.permissions:
            return permission in self.current_user.permissions[file_path]

        # Check directory-based permissions
        for dir_path, permissions in self.current_user.permissions.items():
            if file_path.startswith(dir_path):
                return permission in permissions

        # Role-based default permissions - UPDATED FOR READ-ONLY ACCESS
        if self.current_user.role == UserRole.STANDARD:
            # Standard users can read most files, but write access is very restricted
            if permission == Permission.READ:
                # Can read most directories but not sensitive system files
                if file_path.startswith("/etc/passwd") or file_path.startswith("/etc/shadow") or file_path.startswith("/var/log/auth"):
                    return False
                return True
            elif permission == Permission.WRITE:
                # Only write access to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")
            elif permission == Permission.EXECUTE:
                # Execute access only to their home directory, /tmp, and system binaries
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp") or file_path.startswith("/bin") or file_path.startswith("/usr/bin")

        elif self.current_user.role == UserRole.GUEST:
            # Guests have very limited permissions - mostly read-only
            if permission == Permission.READ:
                # Can read their home directory, /tmp, and some system directories
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp") or file_path.startswith("/home") or file_path.startswith("/usr") or file_path.startswith("/bin")
            elif permission == Permission.WRITE:
                # Only write access to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")
            elif permission == Permission.EXECUTE:
                # Execute access only to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")

        return False

    def get_file_permission_info(self, file_path: str) -> Dict[str, bool]:
        """Get detailed permission information for a file"""
        if not self.current_user:
            return {"read": False, "write": False, "execute": False}
        
        return {
            "read": self.has_permission(file_path, Permission.READ),
            "write": self.has_permission(file_path, Permission.WRITE),
            "execute": self.has_permission(file_path, Permission.EXECUTE)
        }

    def add_user(self, username: str, password: str, role: UserRole, home_directory: str) -> bool:
        """Add a new user (admin only)"""
        if not self.current_user or self.current_user.role != UserRole.ADMIN:
            return False

        if username in self.users:
            return False

        password_hash = self._hash_password(password)
        new_user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            home_directory=home_directory
        )

        self.users[username] = new_user
        self.save_users()
        return True

    def change_password(self, username: str, new_password: str) -> bool:
        """Change user password"""
        if not self.current_user:
            return False

        # Users can only change their own password, or admin can change any
        if self.current_user.username != username and self.current_user.role != UserRole.ADMIN:
            return False

        if username not in self.users:
            return False

        password_hash = self._hash_password(new_password)
        self.users[username].password_hash = password_hash
        self.save_users()
        return True 