"""Generation 2: Comprehensive Security Framework."""

import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    PATH_TRAVERSAL = "path_traversal"
    INJECTION = "injection"
    OVERFLOW = "overflow"
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_LEAK = "data_leak"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'threat_type': self.threat_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'details': self.details or {}
        }


class PathValidator:
    """Secure path validation to prevent traversal attacks."""
    
    def __init__(self, allowed_roots: List[str], allowed_extensions: Optional[Set[str]] = None):
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots]
        self.allowed_extensions = allowed_extensions or {'.json', '.h5', '.h5ad', '.csv', '.txt', '.dat'}
        
    def validate_path(self, path: Union[str, Path]) -> bool:
        """Validate if path is safe to access."""
        try:
            resolved_path = Path(path).resolve()
            
            # Check if path is within allowed roots
            for allowed_root in self.allowed_roots:
                try:
                    resolved_path.relative_to(allowed_root)
                    # Check file extension
                    if resolved_path.suffix.lower() in self.allowed_extensions:
                        return True
                except ValueError:
                    continue
            
            return False
            
        except Exception:
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe usage with enhanced security."""
        # Remove path separators and special characters (more restrictive)
        safe_chars = re.sub(r'[^\w\-_\.]', '', filename)
        
        # Remove leading dots and limit length
        safe_chars = safe_chars.lstrip('.').strip()
        safe_chars = safe_chars[:200]  # More restrictive length limit
        
        # Additional security checks
        dangerous_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 
                          'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3', 
                          'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
        
        if safe_chars.lower().split('.')[0] in dangerous_names:
            safe_chars = f"safe_{safe_chars}"
        
        # Ensure it's not empty and has valid extension
        if not safe_chars or safe_chars == '.' or safe_chars.startswith('..'):
            safe_chars = f"secure_file_{secrets.token_hex(8)}.dat"
        
        return safe_chars


class InputValidator:
    """Comprehensive input validation for security."""
    
    def __init__(self):
        # Enhanced patterns for potential security threats
        self.sql_injection_patterns = [
            r"(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bUNION\b|\bALTER\b|\bCREATE\b)",
            r"(--|\#|\/\*|\*\/|;)",
            r"(\bOR\b|\bAND\b).*[=<>]",
            r"(\bEXEC\b|\bEXECUTE\b)",
            r"(\bSYS\b|\bINFORMATION_SCHEMA\b)",
        ]
        
        self.script_injection_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"document\.",
            r"window\.",
            r"<iframe",
            r"<object",
            r"<embed",
            r"data:text/html",
            r"vbscript:",
        ]
        
        self.path_traversal_patterns = [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"\.\.[\\/]",
            r"\%2e\%2e[\\/]",
            r"\%2f",
            r"\%5c",
            r"\.\.\\x2f",
            r"\.\.\\x5c",
            r"file://",
            r"\\\\",
        ]
        
        # Additional command injection patterns  
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"^\s*\w+\s*=",
            r"(\bcat\b|\bls\b|\brm\b|\bcp\b|\bmv\b)",
            r"(\bcurl\b|\bwget\b|\bpython\b|\bnode\b)",
            r"(\bsudo\b|\bsu\b|\bchmod\b|\bchown\b)",
        ]
        
    def validate_string_input(self, value: str, max_length: int = 1000) -> bool:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            return False
        
        if len(value) > max_length:
            return False
        
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        # Check for script injection patterns
        for pattern in self.script_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        # Check for path traversal patterns
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        # Check for command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def validate_numeric_input(self, value: Union[int, float], 
                             min_val: Optional[float] = None, 
                             max_val: Optional[float] = None) -> bool:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        # Check for overflow
        if abs(value) > 1e10:  # Reasonable limit
            return False
        
        return True
    
    def validate_dict_structure(self, data: Dict[str, Any], 
                              max_depth: int = 10, 
                              max_keys: int = 1000) -> bool:
        """Validate dictionary structure for security."""
        if not isinstance(data, dict):
            return False
        
        def _check_recursive(obj, depth=0):
            if depth > max_depth:
                return False
            
            if isinstance(obj, dict):
                if len(obj) > max_keys:
                    return False
                
                for key, value in obj.items():
                    if not self.validate_string_input(str(key), 100):
                        return False
                    
                    if not _check_recursive(value, depth + 1):
                        return False
            
            elif isinstance(obj, list):
                if len(obj) > max_keys:
                    return False
                
                for item in obj:
                    if not _check_recursive(item, depth + 1):
                        return False
            
            elif isinstance(obj, str):
                if not self.validate_string_input(obj):
                    return False
            
            elif isinstance(obj, (int, float)):
                if not self.validate_numeric_input(obj):
                    return False
            
            return True
        
        return _check_recursive(data)


class AccessControl:
    """Access control and rate limiting."""
    
    def __init__(self):
        self.access_log: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        self.rate_limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'strict': {'requests': 10, 'window': 3600},    # 10 requests per hour
        }
    
    def check_rate_limit(self, identifier: str, limit_type: str = 'default') -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        if identifier in self.blocked_ips:
            return False
        
        if identifier not in self.access_log:
            self.access_log[identifier] = []
        
        # Clean old entries
        limit_config = self.rate_limits.get(limit_type, self.rate_limits['default'])
        window_start = current_time - limit_config['window']
        
        self.access_log[identifier] = [
            timestamp for timestamp in self.access_log[identifier]
            if timestamp > window_start
        ]
        
        # Check if within limits
        if len(self.access_log[identifier]) >= limit_config['requests']:
            # Block if too many requests
            self.blocked_ips.add(identifier)
            return False
        
        # Record current request
        self.access_log[identifier].append(current_time)
        return True
    
    def block_identifier(self, identifier: str) -> None:
        """Block an identifier."""
        self.blocked_ips.add(identifier)
    
    def unblock_identifier(self, identifier: str) -> None:
        """Unblock an identifier."""
        self.blocked_ips.discard(identifier)


class DataEncryption:
    """Simple data encryption for sensitive information."""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return secrets.token_bytes(32)
    
    def hash_data(self, data: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
        """Hash data with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt, 100000)
        return hashed.hex(), salt
    
    def verify_hash(self, data: str, hashed: str, salt: bytes) -> bool:
        """Verify hashed data."""
        try:
            new_hash, _ = self.hash_data(data, salt)
            return hmac.compare_digest(new_hash, hashed)
        except Exception:
            return False
    
    def create_token(self, data: Dict[str, Any], expiry_hours: int = 24) -> str:
        """Create a secure token."""
        token_data = {
            'data': data,
            'created': time.time(),
            'expires': time.time() + (expiry_hours * 3600)
        }
        
        # Simple token creation (in production, use proper JWT)
        token_str = str(token_data)
        token_hash = hmac.new(self.key, token_str.encode(), hashlib.sha256).hexdigest()
        
        return f"{token_hash}:{token_str}"
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and extract token data."""
        try:
            if ':' not in token:
                return None
            
            token_hash, token_str = token.split(':', 1)
            
            # Verify hash
            expected_hash = hmac.new(self.key, token_str.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(token_hash, expected_hash):
                return None
            
            # Parse data
            token_data = eval(token_str)  # In production, use proper JSON parsing
            
            # Check expiry
            if time.time() > token_data['expires']:
                return None
            
            return token_data['data']
            
        except Exception:
            return None


class SecurityAuditor:
    """Security auditing and monitoring."""
    
    def __init__(self, log_file: str = "./logs/security.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[SecurityEvent] = []
        
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.events.append(event)
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{event.to_dict()}\n")
        except Exception:
            pass  # Fail silently for logging
    
    def create_threat_event(self, threat_type: ThreatType, 
                           description: str, 
                           severity: SecurityLevel = SecurityLevel.MEDIUM,
                           **kwargs) -> SecurityEvent:
        """Create and log a threat event."""
        event = SecurityEvent(
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            description=description,
            **kwargs
        )
        
        self.log_security_event(event)
        return event
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        threat_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            threat_counts[event.threat_type.value] = threat_counts.get(event.threat_type.value, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        return {
            'total_events': len(recent_events),
            'threat_counts': threat_counts,
            'severity_counts': severity_counts,
            'high_severity_events': len([e for e in recent_events 
                                       if e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]])
        }


class SecureOperationManager:
    """Manager for secure operations with comprehensive protection."""
    
    def __init__(self, allowed_paths: List[str]):
        self.path_validator = PathValidator(allowed_paths)
        self.input_validator = InputValidator()
        self.access_control = AccessControl()
        self.encryption = DataEncryption()
        self.auditor = SecurityAuditor()
        
    def validate_operation(self, operation: str, data: Any, 
                         identifier: str = "unknown") -> bool:
        """Validate if operation is secure."""
        # Enhanced security: Check rate limits with stricter defaults
        if not self.access_control.check_rate_limit(identifier):
            self.auditor.create_threat_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                f"Rate limit exceeded for {identifier}",
                SecurityLevel.HIGH,
                user_id=identifier
            )
            return False
            
        # Additional security: Check for suspicious operations
        if operation in ['exec', 'eval', 'import', '__import__']:
            self.auditor.create_threat_event(
                ThreatType.MALICIOUS_INPUT,
                f"Potentially dangerous operation blocked: {operation}",
                SecurityLevel.CRITICAL,
                user_id=identifier
            )
            return False
        
        # Validate input data
        if isinstance(data, str):
            if not self.input_validator.validate_string_input(data):
                self.auditor.create_threat_event(
                    ThreatType.MALICIOUS_INPUT,
                    f"Invalid string input detected in {operation}",
                    SecurityLevel.MEDIUM,
                    details={'operation': operation, 'data_type': 'string'}
                )
                return False
        
        elif isinstance(data, dict):
            if not self.input_validator.validate_dict_structure(data):
                self.auditor.create_threat_event(
                    ThreatType.MALICIOUS_INPUT,
                    f"Invalid dict structure detected in {operation}",
                    SecurityLevel.MEDIUM,
                    details={'operation': operation, 'data_type': 'dict'}
                )
                return False
        
        return True
    
    def secure_file_operation(self, filepath: str, operation: str = "read") -> bool:
        """Validate file operation security."""
        if not self.path_validator.validate_path(filepath):
            self.auditor.create_threat_event(
                ThreatType.PATH_TRAVERSAL,
                f"Suspicious file path: {filepath}",
                SecurityLevel.HIGH,
                details={'filepath': filepath, 'operation': operation}
            )
            return False
        
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'blocked_ips': len(self.access_control.blocked_ips),
            'security_summary': self.auditor.get_security_summary(),
            'validation_rules_active': True,
            'encryption_enabled': True
        }


# Global security manager
_global_security_manager: Optional[SecureOperationManager] = None


def get_security_manager(allowed_paths: Optional[List[str]] = None) -> SecureOperationManager:
    """Get the global security manager."""
    global _global_security_manager
    
    if _global_security_manager is None:
        # Enhanced security: More restrictive default paths
        default_paths = ['./data', './output']
        _global_security_manager = SecureOperationManager(allowed_paths or default_paths)
    
    return _global_security_manager


def secure_operation(operation_name: str, allowed_paths: Optional[List[str]] = None):
    """Decorator for secure operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager = get_security_manager(allowed_paths)
            
            # Basic validation
            identifier = kwargs.get('user_id', 'anonymous')
            
            # Validate operation
            if not security_manager.validate_operation(operation_name, args, identifier):
                raise PermissionError(f"Security validation failed for {operation_name}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator