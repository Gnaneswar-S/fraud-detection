"""
JWT Token Blacklist
====================
Allows revoking specific JWT tokens before they expire.
Solves the core weakness of stateless JWT: once issued, a token
is valid until expiry even if the user's account is compromised.

Use cases:
  1. User logs out manually (token immediately invalidated)
  2. Admin revokes a compromised analyst account
  3. Password changed → all existing tokens invalidated
  4. Suspicious activity detected → emergency lockout

Architecture:
  In-process dict for development/single-instance.
  For multi-instance production, replace with Redis:

    import redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    def revoke(jti: str, ttl_seconds: int):
        r.setex(f"blacklist:{jti}", ttl_seconds, "1")

    def is_revoked(jti: str) -> bool:
        return bool(r.exists(f"blacklist:{jti}"))

How it works:
  JWT tokens carry a "jti" (JWT ID) claim — a unique ID per token.
  On revocation, we store the jti in the blacklist with the same TTL
  as the token's remaining validity. The get_current_user() dependency
  checks the blacklist before accepting any token.

  Memory is bounded: entries auto-expire at token expiry time.

Usage:
  # In auth.py — add jti to token payload:
  import uuid
  data["jti"] = str(uuid.uuid4())

  # In get_current_user() — check blacklist:
  from api.token_blacklist import blacklist
  jti = payload.get("jti")
  if jti and blacklist.is_revoked(jti):
      raise credentials_exception

  # On logout:
  from api.token_blacklist import blacklist
  blacklist.revoke(jti, remaining_seconds)
"""
from __future__ import annotations

import time
import threading
from typing import Optional


class TokenBlacklist:
    """
    Thread-safe in-process JWT token blacklist with TTL.

    Stores {jti: expiry_timestamp} pairs.
    Expired entries are cleaned up automatically.
    """

    def __init__(self):
        self._store: dict[str, float] = {}   # jti → unix expiry time
        self._lock  = threading.Lock()
        self._start_cleanup()

    def revoke(self, jti: str, ttl_seconds: int = 3600) -> None:
        """
        Add a token ID to the blacklist.

        Parameters
        ----------
        jti         JWT ID claim from the token payload
        ttl_seconds How long to keep the blacklist entry (should match
                    the token's remaining lifetime)
        """
        expiry = time.time() + ttl_seconds
        with self._lock:
            self._store[jti] = expiry

    def is_revoked(self, jti: Optional[str]) -> bool:
        """
        Return True if this jti has been revoked AND the entry
        has not yet expired (i.e. the token is still in its validity
        window but explicitly blocked).
        """
        if not jti:
            return False
        with self._lock:
            expiry = self._store.get(jti)
            if expiry is None:
                return False
            if time.time() > expiry:
                del self._store[jti]   # lazy cleanup
                return False
            return True

    def revoke_all_for_user(self, username: str, active_jtis: list[str],
                            ttl_seconds: int = 3600) -> int:
        """
        Revoke all active tokens for a given user (e.g. on password change).
        Caller must provide the list of active JTIs for that user.
        Returns number of tokens revoked.
        """
        count = 0
        for jti in active_jtis:
            self.revoke(jti, ttl_seconds)
            count += 1
        return count

    def size(self) -> int:
        """Number of active blacklist entries."""
        with self._lock:
            now = time.time()
            return sum(1 for exp in self._store.values() if exp > now)

    def _start_cleanup(self):
        """Periodically remove expired entries."""
        def _run():
            while True:
                time.sleep(300)
                now = time.time()
                with self._lock:
                    expired = [jti for jti, exp in self._store.items() if exp <= now]
                    for jti in expired:
                        del self._store[jti]
        t = threading.Thread(target=_run, daemon=True)
        t.start()


# Singleton
blacklist = TokenBlacklist()