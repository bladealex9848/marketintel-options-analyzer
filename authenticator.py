"""
Core authentication functionality for the trading platform.
Implements secure password validation and session management.
"""

import streamlit as st
from datetime import datetime, timedelta
import hashlib
import logging
from typing import Optional, Dict

class InvalidAccessError(Exception):
    """Exception raised for invalid authentication attempts."""
    pass

def _get_valid_passwords() -> set:
    """
    Retrieve valid password hashes from Streamlit secrets.
    
    Returns:
        set: Set of valid password hashes
    """
    try:
        # Obtener passwords del secrets.toml
        passwords = st.secrets.get("auth", {}).get("valid_passwords", [])
        # Convertir passwords a hashes
        return {hashlib.sha256(pwd.encode()).hexdigest() for pwd in passwords}
    except Exception as e:
        logging.error(f"Error accessing secrets: {e}")
        return set()

def _hash_password(password: str) -> str:
    """
    Hash a password using SHA-256.
    
    Args:
        password (str): Password to hash
        
    Returns:
        str: Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str) -> bool:
    """
    Validate password and manage authentication attempts.
    
    Args:
        password (str): Password to validate
        
    Returns:
        bool: True if password is valid, False otherwise
    """
    if not password:
        return False
        
    # Inicializar contador de intentos
    if 'auth_attempts' not in st.session_state:
        st.session_state.auth_attempts = 0
        st.session_state.last_attempt = datetime.now()
    
    # Verificar bloqueo temporal
    if st.session_state.auth_attempts >= 3:
        time_diff = datetime.now() - st.session_state.last_attempt
        if time_diff < timedelta(minutes=15):
            remaining = 15 - (time_diff.seconds // 60)
            st.error(f"Demasiados intentos. Espere {remaining} minutos.")
            return False
        else:
            st.session_state.auth_attempts = 0
    
    # Validar password
    password_hash = _hash_password(password)
    valid_passwords = _get_valid_passwords()
    
    if password_hash in valid_passwords:
        st.session_state.auth_attempts = 0
        st.session_state.last_successful_auth = datetime.now()
        return True
    else:
        st.session_state.auth_attempts += 1
        st.session_state.last_attempt = datetime.now()
        attempts_left = 3 - st.session_state.auth_attempts
        if attempts_left > 0:
            st.error(f"Password inválido. {attempts_left} intentos restantes.")
        return False

def validate_session() -> bool:
    """
    Validate current session is authentic and not expired.
    
    Returns:
        bool: True if session is valid, False otherwise
    """
    if not st.session_state.get("authenticated", False):
        return False
        
    last_auth = st.session_state.get("last_successful_auth")
    if not last_auth or datetime.now() - last_auth > timedelta(hours=8):
        st.session_state.authenticated = False
        return False
        
    return True

def get_session_info() -> Dict:
    """
    Get current session information.
    
    Returns:
        Dict: Session information including authentication status and timestamp
    """
    return {
        "authenticated": st.session_state.get("authenticated", False),
        "last_auth": st.session_state.get("last_successful_auth"),
        "attempts": st.session_state.get("auth_attempts", 0)
    }

def clear_session():
    """Clear all authentication session data."""
    if "authenticated" in st.session_state:
        del st.session_state.authenticated
    if "auth_attempts" in st.session_state:
        del st.session_state.auth_attempts
    if "last_attempt" in st.session_state:
        del st.session_state.last_attempt
    if "last_successful_auth" in st.session_state:
        del st.session_state.last_successful_auth