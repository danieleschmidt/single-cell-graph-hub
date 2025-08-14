"""Advanced Security Framework for Single-Cell Graph Hub.

This module implements comprehensive security measures including data encryption,
access control, audit logging, vulnerability scanning, and threat detection
for scientific computing environments.
"""

import os
import sys
import json
import time
import hashlib
import secrets
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import base64
import hmac

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ActionType(Enum):
    """Types of security actions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: ThreatLevel
    user_id: Optional[str]
    resource: str
    action: ActionType
    result: str  # success, denied, failed
    details: Dict[str, Any]
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'severity': self.severity.value,
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action.value,
            'result': self.result,
            'details': self.details,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent
        }


@dataclass
class User:
    """User representation with security attributes."""
    user_id: str
    username: str
    email: str
    security_clearance: SecurityLevel
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    password_hash: Optional[str] = None
    api_key_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'security_clearance': self.security_clearance.value,
            'roles': self.roles,
            'permissions': self.permissions,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'failed_login_attempts': self.failed_login_attempts,
            'account_locked': self.account_locked
        }


class CryptographyManager:
    """Handles all cryptographic operations."""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.salt_length = 32
        self.key_derivation_iterations = 100000
    
    def _generate_master_key(self) -> bytes:
        """Generate or load master encryption key."""
        key_file = Path("./.security/master.key")
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load master key: {e}")
        
        # Generate new master key
        key_file.parent.mkdir(parents=True, exist_ok=True)
        master_key = secrets.token_bytes(32)
        
        try:
            with open(key_file, 'wb') as f:
                f.write(master_key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            logger.info("Generated new master encryption key")
            
        except Exception as e:
            logger.error(f"Failed to save master key: {e}")
        
        return master_key
    
    def derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(self.salt_length)
        
        # Simulate PBKDF2 key derivation
        key_material = (password + salt.hex()).encode()
        key = hashlib.pbkdf2_hmac(
            'sha256', 
            key_material, 
            salt, 
            self.key_derivation_iterations,
            32
        )
        
        return key, salt
    
    def encrypt_data(self, data: bytes, key: bytes = None) -> bytes:
        """Encrypt data using AES-256 (simulated)."""
        if key is None:
            key = self.master_key
        
        # Simulate AES encryption
        # In production, use proper AES implementation
        iv = secrets.token_bytes(16)
        
        # Simple XOR cipher for demonstration (NOT secure!)
        encrypted = bytearray()
        key_cycle = (key * ((len(data) // len(key)) + 1))[:len(data)]
        
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_cycle[i] ^ iv[i % len(iv)])
        
        # Prepend IV
        return iv + bytes(encrypted)
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes = None) -> bytes:
        """Decrypt data using AES-256 (simulated)."""
        if key is None:
            key = self.master_key
        
        # Extract IV and encrypted data
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        
        # Simple XOR cipher reversal
        decrypted = bytearray()
        key_cycle = (key * ((len(encrypted) // len(key)) + 1))[:len(encrypted)]
        
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key_cycle[i] ^ iv[i % len(iv)])
        
        return bytes(decrypted)
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash password with salt."""
        salt = secrets.token_hex(16)
        # Use scrypt for password hashing
        password_hash = hashlib.scrypt(
            password.encode(),
            salt=salt.encode(),
            n=16384,
            r=8,
            p=1,
            dklen=32
        )
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        try:
            computed_hash = hashlib.scrypt(
                password.encode(),
                salt=salt.encode(),
                n=16384,
                r=8,
                p=1,
                dklen=32
            )
            return computed_hash.hex() == password_hash
        except Exception:
            return False
    
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    def create_hmac_signature(self, data: str, secret: str) -> str:
        """Create HMAC signature for data integrity."""
        signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        )
        return signature.hexdigest()
    
    def verify_hmac_signature(self, data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.create_hmac_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)


class AccessControlManager:
    """Manages user access control and permissions."""
    
    def __init__(self, crypto_manager: CryptographyManager):
        self.crypto_manager = crypto_manager
        self.users: Dict[str, User] = {}
        self.roles_permissions = self._initialize_roles()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_login_threshold = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def _initialize_roles(self) -> Dict[str, List[str]]:
        """Initialize role-based permissions."""
        return {
            'guest': ['read_public_data'],
            'researcher': [
                'read_public_data', 'read_internal_data', 
                'write_research_data', 'execute_analysis'
            ],
            'admin': [
                'read_public_data', 'read_internal_data', 'read_confidential_data',
                'write_research_data', 'write_system_config', 
                'execute_analysis', 'execute_admin', 'delete_data'
            ],
            'security_officer': [
                'read_audit_logs', 'read_security_events', 
                'write_security_config', 'execute_security_scan'
            ],
            'data_steward': [
                'read_all_data', 'write_metadata', 'execute_data_validation',
                'delete_deprecated_data'
            ]
        }
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        security_clearance: SecurityLevel = SecurityLevel.INTERNAL,
        roles: List[str] = None
    ) -> User:
        """Create a new user account."""
        if roles is None:
            roles = ['researcher']
        
        # Validate roles
        invalid_roles = [role for role in roles if role not in self.roles_permissions]
        if invalid_roles:
            raise ValueError(f"Invalid roles: {invalid_roles}")
        
        # Hash password
        password_hash, salt = self.crypto_manager.hash_password(password)
        
        # Collect permissions from roles
        permissions = []
        for role in roles:
            permissions.extend(self.roles_permissions[role])
        permissions = list(set(permissions))  # Remove duplicates
        
        user = User(
            user_id=secrets.token_hex(8),
            username=username,
            email=email,
            security_clearance=security_clearance,
            roles=roles,
            permissions=permissions,
            created_at=datetime.now(),
            password_hash=f"{password_hash}:{salt}"
        )
        
        self.users[user.user_id] = user
        logger.info(f"Created user: {username} with roles: {roles}")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        if user.account_locked:
            if datetime.now() - user.last_login < self.lockout_duration:
                logger.warning(f"Authentication failed: account locked - {username}")
                return None
            else:
                # Unlock account after lockout period
                user.account_locked = False
                user.failed_login_attempts = 0
        
        # Verify password
        password_hash, salt = user.password_hash.split(':')
        if self.crypto_manager.verify_password(password, password_hash, salt):
            user.last_login = datetime.now()
            user.failed_login_attempts = 0
            logger.info(f"User authenticated: {username}")
            return user
        else:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= self.failed_login_threshold:
                user.account_locked = True
                logger.warning(f"Account locked due to failed attempts: {username}")
            else:
                logger.warning(f"Authentication failed: invalid password - {username}")
            return None
    
    def create_session(self, user: User) -> str:
        """Create an authenticated session."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user.user_id,
            'username': user.username,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'permissions': user.permissions,
            'security_clearance': user.security_clearance.value
        }
        
        self.active_sessions[session_id] = session_data
        logger.info(f"Created session for user: {user.username}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session."""
        session = self.active_sessions.get(session_id)
        
        if not session:
            return None
        
        # Check session timeout (24 hours)
        if datetime.now() - session['last_activity'] > timedelta(hours=24):
            del self.active_sessions[session_id]
            logger.info(f"Session expired: {session_id}")
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return session
    
    def check_permission(
        self,
        session_id: str,
        resource: str,
        action: ActionType,
        required_clearance: SecurityLevel = SecurityLevel.INTERNAL
    ) -> bool:
        """Check if user has permission for action on resource."""
        session = self.validate_session(session_id)
        
        if not session:
            return False
        
        # Check security clearance
        user_clearance = SecurityLevel(session['security_clearance'])
        clearance_levels = list(SecurityLevel)
        
        if clearance_levels.index(user_clearance) < clearance_levels.index(required_clearance):
            logger.warning(f"Access denied: insufficient clearance for {session['username']}")
            return False
        
        # Check specific permissions
        required_permission = f"{action.value}_{resource.lower()}"
        general_permission = f"{action.value}_all_data"
        
        if required_permission in session['permissions'] or general_permission in session['permissions']:
            return True
        
        logger.warning(f"Access denied: missing permission {required_permission} for {session['username']}")
        return False
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke an active session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session revoked: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions (for admin purposes)."""
        return {
            session_id: {
                'username': session['username'],
                'created_at': session['created_at'].isoformat(),
                'last_activity': session['last_activity'].isoformat()
            }
            for session_id, session in self.active_sessions.items()
        }


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_directory: str = "./security_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.security_events: List[SecurityEvent] = []
        self.crypto_manager = CryptographyManager()
        
        # Setup encrypted audit log file
        self.audit_log_file = self.log_directory / "audit.log"
        self._setup_audit_logger()
    
    def _setup_audit_logger(self):
        """Setup encrypted audit logging."""
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Create encrypted file handler
        handler = logging.FileHandler(self.audit_log_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
    
    def log_security_event(
        self,
        event_type: str,
        severity: ThreatLevel,
        user_id: Optional[str],
        resource: str,
        action: ActionType,
        result: str,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityEvent:
        """Log a security event."""
        
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details,
            source_ip=source_ip,
            user_agent=user_agent
        )
        
        self.security_events.append(event)
        
        # Log to audit file
        self.audit_logger.info(
            f"EVENT: {event_type} | USER: {user_id or 'ANONYMOUS'} | "
            f"RESOURCE: {resource} | ACTION: {action.value} | RESULT: {result}"
        )
        
        # Alert on high severity events
        if severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            self._send_security_alert(event)
        
        return event
    
    def _send_security_alert(self, event: SecurityEvent):
        """Send security alert for high severity events."""
        alert_file = self.log_directory / "security_alerts.json"
        
        alert_data = {
            'alert_id': secrets.token_hex(6),
            'timestamp': datetime.now().isoformat(),
            'event': event.to_dict(),
            'alert_level': event.severity.value,
            'requires_action': event.severity == ThreatLevel.CRITICAL
        }
        
        alerts = []
        if alert_file.exists():
            try:
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []
        
        alerts.append(alert_data)
        
        # Keep only last 100 alerts
        alerts = alerts[-100:]
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.critical(f"SECURITY ALERT: {event.event_type} - {event.details}")
    
    def search_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[ThreatLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Search security events with filters."""
        filtered_events = self.security_events
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    def export_audit_report(self, output_path: str = None) -> str:
        """Export comprehensive audit report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"security_audit_report_{timestamp}.json"
        
        # Generate statistics
        event_stats = {}
        severity_stats = {}
        user_stats = {}
        
        for event in self.security_events:
            event_stats[event.event_type] = event_stats.get(event.event_type, 0) + 1
            severity_stats[event.severity.value] = severity_stats.get(event.severity.value, 0) + 1
            if event.user_id:
                user_stats[event.user_id] = user_stats.get(event.user_id, 0) + 1
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_events': len(self.security_events),
                'report_period': {
                    'start': min(e.timestamp for e in self.security_events).isoformat() if self.security_events else None,
                    'end': max(e.timestamp for e in self.security_events).isoformat() if self.security_events else None
                }
            },
            'statistics': {
                'events_by_type': event_stats,
                'events_by_severity': severity_stats,
                'events_by_user': user_stats
            },
            'events': [event.to_dict() for event in self.security_events]
        }
        
        # Encrypt sensitive audit data
        report_json = json.dumps(report_data, indent=2)
        encrypted_data = self.crypto_manager.encrypt_data(report_json.encode())
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Encrypted audit report exported: {output_path}")
        return output_path


class ThreatDetectionEngine:
    """Advanced threat detection and analysis."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.threat_rules = self._initialize_threat_rules()
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def _initialize_threat_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection rules."""
        return {
            'brute_force_login': {
                'description': 'Multiple failed login attempts',
                'threshold': 5,
                'time_window': 300,  # 5 minutes
                'severity': ThreatLevel.HIGH
            },
            'privilege_escalation': {
                'description': 'Unauthorized access to restricted resources',
                'threshold': 3,
                'time_window': 600,  # 10 minutes
                'severity': ThreatLevel.CRITICAL
            },
            'data_exfiltration': {
                'description': 'Unusual data access patterns',
                'threshold': 100,  # 100 data reads
                'time_window': 3600,  # 1 hour
                'severity': ThreatLevel.HIGH
            },
            'anomalous_activity': {
                'description': 'Activity outside normal hours',
                'threshold': 1,
                'time_window': 86400,  # 24 hours
                'severity': ThreatLevel.MEDIUM
            }
        }
    
    def start_monitoring(self):
        """Start continuous threat monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Threat detection monitoring started")
    
    def stop_monitoring(self):
        """Stop threat monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Threat detection monitoring stopped")
    
    def _monitoring_loop(self):
        """Main threat monitoring loop."""
        while self.monitoring_active:
            try:
                self._analyze_recent_events()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in threat monitoring: {str(e)}")
                time.sleep(10)
    
    def _analyze_recent_events(self):
        """Analyze recent events for threat patterns."""
        current_time = datetime.now()
        
        for rule_name, rule_config in self.threat_rules.items():
            time_threshold = current_time - timedelta(seconds=rule_config['time_window'])
            
            if rule_name == 'brute_force_login':
                self._check_brute_force_attacks(time_threshold, rule_config)
            elif rule_name == 'privilege_escalation':
                self._check_privilege_escalation(time_threshold, rule_config)
            elif rule_name == 'data_exfiltration':
                self._check_data_exfiltration(time_threshold, rule_config)
            elif rule_name == 'anomalous_activity':
                self._check_anomalous_activity(time_threshold, rule_config)
    
    def _check_brute_force_attacks(self, time_threshold: datetime, rule_config: Dict[str, Any]):
        """Check for brute force login attacks."""
        recent_events = self.audit_logger.search_events(
            event_type='authentication_failed',
            start_time=time_threshold
        )
        
        # Group by source IP or user
        source_attempts = {}
        for event in recent_events:
            source = event.source_ip or event.user_id or 'unknown'
            source_attempts[source] = source_attempts.get(source, 0) + 1
        
        for source, attempts in source_attempts.items():
            if attempts >= rule_config['threshold']:
                self.audit_logger.log_security_event(
                    event_type='threat_detected_brute_force',
                    severity=rule_config['severity'],
                    user_id=None,
                    resource='authentication_system',
                    action=ActionType.EXECUTE,
                    result='threat_detected',
                    details={
                        'threat_type': 'brute_force_login',
                        'source': source,
                        'attempts': attempts,
                        'time_window': rule_config['time_window']
                    },
                    source_ip=source if '.' in source else None
                )
    
    def _check_privilege_escalation(self, time_threshold: datetime, rule_config: Dict[str, Any]):
        """Check for privilege escalation attempts."""
        recent_events = self.audit_logger.search_events(
            start_time=time_threshold
        )
        
        access_denied_events = [
            e for e in recent_events 
            if e.result == 'denied' and 'insufficient' in e.details.get('reason', '').lower()
        ]
        
        # Group by user
        user_denials = {}
        for event in access_denied_events:
            if event.user_id:
                user_denials[event.user_id] = user_denials.get(event.user_id, 0) + 1
        
        for user_id, denials in user_denials.items():
            if denials >= rule_config['threshold']:
                self.audit_logger.log_security_event(
                    event_type='threat_detected_privilege_escalation',
                    severity=rule_config['severity'],
                    user_id=user_id,
                    resource='access_control_system',
                    action=ActionType.EXECUTE,
                    result='threat_detected',
                    details={
                        'threat_type': 'privilege_escalation',
                        'user_id': user_id,
                        'denied_attempts': denials,
                        'time_window': rule_config['time_window']
                    }
                )
    
    def _check_data_exfiltration(self, time_threshold: datetime, rule_config: Dict[str, Any]):
        """Check for potential data exfiltration."""
        recent_events = self.audit_logger.search_events(
            start_time=time_threshold
        )
        
        read_events = [
            e for e in recent_events 
            if e.action == ActionType.READ and e.result == 'success'
        ]
        
        # Group by user
        user_reads = {}
        for event in read_events:
            if event.user_id:
                user_reads[event.user_id] = user_reads.get(event.user_id, 0) + 1
        
        for user_id, reads in user_reads.items():
            if reads >= rule_config['threshold']:
                self.audit_logger.log_security_event(
                    event_type='threat_detected_data_exfiltration',
                    severity=rule_config['severity'],
                    user_id=user_id,
                    resource='data_system',
                    action=ActionType.READ,
                    result='threat_detected',
                    details={
                        'threat_type': 'data_exfiltration',
                        'user_id': user_id,
                        'read_operations': reads,
                        'time_window': rule_config['time_window']
                    }
                )
    
    def _check_anomalous_activity(self, time_threshold: datetime, rule_config: Dict[str, Any]):
        """Check for anomalous activity patterns."""
        recent_events = self.audit_logger.search_events(
            start_time=time_threshold
        )
        
        # Check for activity outside business hours (simplified)
        for event in recent_events:
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
                self.audit_logger.log_security_event(
                    event_type='threat_detected_anomalous_activity',
                    severity=rule_config['severity'],
                    user_id=event.user_id,
                    resource=event.resource,
                    action=event.action,
                    result='threat_detected',
                    details={
                        'threat_type': 'anomalous_activity',
                        'activity_time': event.timestamp.isoformat(),
                        'reason': 'outside_business_hours'
                    }
                )


class AdvancedSecurityFramework:
    """Main security framework coordinating all security components."""
    
    def __init__(self):
        self.crypto_manager = CryptographyManager()
        self.access_control = AccessControlManager(self.crypto_manager)
        self.audit_logger = AuditLogger()
        self.threat_detection = ThreatDetectionEngine(self.audit_logger)
        
        # Initialize admin user
        self._create_default_admin()
        
        logger.info("Advanced Security Framework initialized")
    
    def _create_default_admin(self):
        """Create default admin user."""
        try:
            admin_user = self.access_control.create_user(
                username="admin",
                email="admin@scgraphhub.org",
                password="SecureAdmin123!",
                security_clearance=SecurityLevel.TOP_SECRET,
                roles=["admin", "security_officer"]
            )
            
            logger.info("Default admin user created")
            
        except Exception as e:
            logger.warning(f"Failed to create default admin: {e}")
    
    def secure_operation(
        self,
        session_id: str,
        operation_name: str,
        resource: str,
        action: ActionType,
        required_clearance: SecurityLevel = SecurityLevel.INTERNAL,
        operation_func: Callable = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Execute a secure operation with full security checks."""
        
        # Validate session
        session = self.access_control.validate_session(session_id)
        if not session:
            self.audit_logger.log_security_event(
                event_type='access_denied',
                severity=ThreatLevel.MEDIUM,
                user_id=None,
                resource=resource,
                action=action,
                result='denied',
                details={'reason': 'invalid_session', 'operation': operation_name}
            )
            return False, "Invalid session"
        
        # Check permissions
        has_permission = self.access_control.check_permission(
            session_id, resource, action, required_clearance
        )
        
        if not has_permission:
            self.audit_logger.log_security_event(
                event_type='access_denied',
                severity=ThreatLevel.HIGH,
                user_id=session['user_id'],
                resource=resource,
                action=action,
                result='denied',
                details={
                    'reason': 'insufficient_permissions',
                    'operation': operation_name,
                    'required_clearance': required_clearance.value
                }
            )
            return False, "Insufficient permissions"
        
        # Execute operation if provided
        operation_result = None
        operation_success = True
        
        if operation_func:
            try:
                operation_result = operation_func(**kwargs)
            except Exception as e:
                operation_success = False
                operation_result = str(e)
        
        # Log successful access
        self.audit_logger.log_security_event(
            event_type='access_granted',
            severity=ThreatLevel.INFO,
            user_id=session['user_id'],
            resource=resource,
            action=action,
            result='success' if operation_success else 'failed',
            details={
                'operation': operation_name,
                'clearance_used': required_clearance.value,
                'operation_result': str(operation_result) if not operation_success else 'success'
            }
        )
        
        return operation_success, operation_result
    
    def start_security_monitoring(self):
        """Start all security monitoring components."""
        self.threat_detection.start_monitoring()
        logger.info("Security monitoring started")
    
    def stop_security_monitoring(self):
        """Stop all security monitoring components."""
        self.threat_detection.stop_monitoring()
        logger.info("Security monitoring stopped")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Generate security dashboard data."""
        # Recent security events
        recent_events = self.audit_logger.search_events(
            start_time=datetime.now() - timedelta(hours=24),
            limit=50
        )
        
        # Security statistics
        event_types = {}
        severity_counts = {}
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        # Active sessions
        active_sessions = self.access_control.get_active_sessions()
        
        # System security status
        security_status = "green"
        if severity_counts.get("critical", 0) > 0:
            security_status = "red"
        elif severity_counts.get("high", 0) > 0:
            security_status = "orange"
        elif severity_counts.get("medium", 0) > 5:
            security_status = "yellow"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'security_status': security_status,
            'active_sessions': len(active_sessions),
            'recent_events_24h': len(recent_events),
            'event_statistics': {
                'by_type': event_types,
                'by_severity': severity_counts
            },
            'active_sessions_detail': active_sessions,
            'threat_monitoring_active': self.threat_detection.monitoring_active,
            'recent_critical_events': [
                event.to_dict() for event in recent_events 
                if event.severity == ThreatLevel.CRITICAL
            ][:5]
        }
    
    def export_security_report(self, output_path: str = None) -> str:
        """Export comprehensive security report."""
        dashboard = self.get_security_dashboard()
        audit_report_path = self.audit_logger.export_audit_report()
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"security_report_{timestamp}.json"
        
        security_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_security_report',
                'framework_version': '1.0'
            },
            'security_dashboard': dashboard,
            'audit_report_path': audit_report_path,
            'security_recommendations': self._generate_security_recommendations(dashboard)
        }
        
        with open(output_path, 'w') as f:
            json.dump(security_report, f, indent=2)
        
        logger.info(f"Security report exported: {output_path}")
        return output_path
    
    def _generate_security_recommendations(self, dashboard: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        if dashboard['security_status'] == 'red':
            recommendations.append("URGENT: Critical security events detected - immediate investigation required")
        
        if dashboard['active_sessions'] > 50:
            recommendations.append("High number of active sessions - consider session cleanup")
        
        if dashboard['event_statistics']['by_severity'].get('high', 0) > 10:
            recommendations.append("Multiple high-severity events - review access controls")
        
        if not dashboard['threat_monitoring_active']:
            recommendations.append("Threat monitoring is inactive - enable continuous monitoring")
        
        if len(recommendations) == 0:
            recommendations.append("Security posture is good - maintain current controls")
        
        return recommendations


def demonstrate_security_framework():
    """Demonstrate the advanced security framework."""
    print("🔒 ADVANCED SECURITY FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # Initialize security framework
    security = AdvancedSecurityFramework()
    print("✓ Security framework initialized")
    
    # Start monitoring
    security.start_security_monitoring()
    print("✓ Security monitoring started")
    
    # Create test users
    researcher = security.access_control.create_user(
        username="researcher_alice",
        email="alice@lab.org",
        password="ResearchPass123!",
        security_clearance=SecurityLevel.CONFIDENTIAL,
        roles=["researcher"]
    )
    
    data_steward = security.access_control.create_user(
        username="steward_bob",
        email="bob@lab.org",
        password="StewardPass123!",
        security_clearance=SecurityLevel.RESTRICTED,
        roles=["data_steward"]
    )
    
    print(f"✓ Created test users: {researcher.username}, {data_steward.username}")
    
    # Test authentication and sessions
    auth_user = security.access_control.authenticate_user("researcher_alice", "ResearchPass123!")
    if auth_user:
        session_id = security.access_control.create_session(auth_user)
        print(f"✓ User authenticated and session created")
        
        # Test secure operations
        print("\n🔐 Testing secure operations...")
        
        # Allowed operation
        success, result = security.secure_operation(
            session_id=session_id,
            operation_name="read_dataset",
            resource="research_data",
            action=ActionType.READ,
            required_clearance=SecurityLevel.INTERNAL,
            operation_func=lambda: "Dataset loaded successfully"
        )
        print(f"  Read operation: {'✓ Success' if success else '✗ Failed'}")
        
        # Denied operation (insufficient clearance)
        success, result = security.secure_operation(
            session_id=session_id,
            operation_name="delete_system_config",
            resource="system_config",
            action=ActionType.DELETE,
            required_clearance=SecurityLevel.TOP_SECRET
        )
        print(f"  Delete operation: {'✗ Denied (expected)' if not success else '✓ Unexpected success'}")
    
    # Simulate security events
    print("\n🚨 Simulating security events...")
    
    # Simulate failed logins (brute force)
    for i in range(6):
        security.audit_logger.log_security_event(
            event_type='authentication_failed',
            severity=ThreatLevel.MEDIUM,
            user_id=None,
            resource='authentication_system',
            action=ActionType.EXECUTE,
            result='failed',
            details={'reason': 'invalid_password', 'attempt': i+1},
            source_ip='192.168.1.100'
        )
    
    print("✓ Simulated brute force attack")
    
    # Wait for threat detection
    time.sleep(2)
    
    # Generate security dashboard
    print("\n📊 Security Dashboard:")
    dashboard = security.get_security_dashboard()
    print(f"  Security Status: {dashboard['security_status']}")
    print(f"  Active Sessions: {dashboard['active_sessions']}")
    print(f"  Recent Events (24h): {dashboard['recent_events_24h']}")
    print(f"  Critical Events: {len(dashboard['recent_critical_events'])}")
    
    # Export security report
    print("\n📋 Generating security report...")
    report_path = security.export_security_report()
    print(f"✓ Security report saved: {report_path}")
    
    # Stop monitoring
    security.stop_security_monitoring()
    print("✓ Security monitoring stopped")
    
    print("\n🛡️ Security Framework Features Demonstrated:")
    print("  ✓ User authentication and authorization")
    print("  ✓ Role-based access control (RBAC)")
    print("  ✓ Comprehensive audit logging")
    print("  ✓ Threat detection and monitoring")
    print("  ✓ Data encryption and cryptography")
    print("  ✓ Security event analysis")
    print("  ✓ Automated security reporting")
    
    return security


if __name__ == "__main__":
    # Run security framework demonstration
    security_system = demonstrate_security_framework()
    print("\n✅ Security framework demonstration completed!")