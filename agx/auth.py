"""
AG-X 2026 Authentication Module
================================

JWT-based authentication and authorization system with password hashing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel, EmailStr

from agx.config import AuthConfig


# =============================================================================
# Schemas
# =============================================================================

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    username_or_email: str
    password: str


class Token(BaseModel):
    """JWT token schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    user_id: int
    username: str
    role: str


# =============================================================================
# Password Hashing
# =============================================================================

class PasswordHasher:
    """Secure password hashing using bcrypt."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password from database
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


# =============================================================================
# JWT Token Management
# =============================================================================

class TokenManager:
    """JWT token creation and validation."""
    
    def __init__(self, config: AuthConfig):
        """Initialize token manager with configuration.
        
        Args:
            config: Authentication configuration
        """
        self.config = config
        self.secret_key = config.secret_key
        self.algorithm = config.algorithm
        self.access_token_expires = timedelta(minutes=config.access_token_expire_minutes)
        self.refresh_token_expires = timedelta(days=config.refresh_token_expire_days)
    
    def create_access_token(self, user_id: int, username: str, role: str) -> str:
        """Create a JWT access token.
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            
        Returns:
            Encoded JWT access token
        """
        expires = datetime.utcnow() + self.access_token_expires
        
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "type": "access",
            "exp": expires,
            "iat": datetime.utcnow(),
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: int) -> str:
        """Create a JWT refresh token.
        
        Args:
            user_id: User ID
            
        Returns:
            Encoded JWT refresh token
        """
        expires = datetime.utcnow() + self.refresh_token_expires
        
        to_encode = {
            "sub": str(user_id),
            "type": "refresh",
            "exp": expires,
            "iat": datetime.utcnow(),
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_tokens(self, user_id: int, username: str, role: str) -> Token:
        """Create both access and refresh tokens.
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            
        Returns:
            Token object with access and refresh tokens
        """
        access_token = self.create_access_token(user_id, username, role)
        refresh_token = self.create_refresh_token(user_id)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
        )
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user_id: int = int(payload.get("sub"))
            username: str = payload.get("username")
            role: str = payload.get("role")
            
            if user_id is None:
                return None
            
            return TokenData(user_id=user_id, username=username or "", role=role or "student")
        
        except JWTError as e:
            logger.warning(f"JWT verification error: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None


# =============================================================================
# Password Validation
# =============================================================================

class PasswordValidator:
    """Validate password strength according to configuration."""
    
    def __init__(self, config: AuthConfig):
        """Initialize password validator with configuration.
        
        Args:
            config: Authentication configuration
        """
        self.config = config
    
    def validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < self.config.min_password_length:
            return False, f"Password must be at least {self.config.min_password_length} characters long"
        
        if self.config.require_number and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        if self.config.require_special_char:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False, "Password must contain at least one special character"
        
        return True, ""


# =============================================================================
# Authentication Service
# =============================================================================

class AuthService:
    """Main authentication service combining all auth functionality."""
    
    def __init__(self, config: AuthConfig):
        """Initialize authentication service.
        
        Args:
            config: Authentication configuration
        """
        self.config = config
        self.password_hasher = PasswordHasher()
        self.token_manager = TokenManager(config)
        self.password_validator = PasswordValidator(config)
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.password_validator.validate_password(password)
    
    def hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.password_hasher.hash_password(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password from database
            
        Returns:
            True if password matches
        """
        return self.password_hasher.verify_password(plain_password, hashed_password)
    
    def create_tokens(self, user_id: int, username: str, role: str) -> Token:
        """Create authentication tokens for a user.
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            
        Returns:
            Token object with access and refresh tokens
        """
        return self.token_manager.create_tokens(user_id, username, role)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify an authentication token.
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None otherwise
        """
        return self.token_manager.verify_token(token)
