"""The one place a request's scope becomes a store predicate.

Files and their chunks are user-scoped and carry raw content, so scope is
enforced structurally rather than remembered by each caller. Every retrieval
path — ``/query``, ``/query_multiple`` and the ``fast-v1`` stored-vector
lookup — builds its predicate from :class:`ScopeFilter`. Drift between
hand-written scope clauses is the realistic failure mode; one builder makes it
impossible rather than reviewable.

The predicate is always ``(tenant, owner/entity, file)`` and always goes into
the store query *before* ranking. Nothing downstream re-derives authorization
from what the store returned.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException, Request, status

from app.auth import BASE_TENANT_ID, SYSTEM_TENANT_ID, Principal
from app.config import logger

PUBLIC_OWNER = "public"

TENANT_METADATA_KEY = "tenant_id"
OWNER_METADATA_KEY = "user_id"


@dataclass(frozen=True)
class ScopeFilter:
    """The tenant and owners a request may read, as store-query clauses."""

    tenant: str
    owners: Tuple[str, ...]

    def tenant_values(self) -> List[Optional[str]]:
        """Tenant values that satisfy this scope.

        Chunks written before tenants were recorded carry no ``tenant_id`` at
        all. They normalize to the base tenant — the same absent/null → base
        rule the projector applies — so the base tenant matches them and a real
        tenant never absorbs untagged content. ``None`` in an ``$in`` list means
        "or the key is absent", matching MongoDB's own ``$in: [null]``
        semantics; the pgvector filter builder implements the same rule.
        """
        if self.tenant == BASE_TENANT_ID:
            return [BASE_TENANT_ID, None]
        return [self.tenant]

    def scope_clauses(self) -> List[Dict[str, Any]]:
        return [
            {OWNER_METADATA_KEY: {"$in": list(self.owners)}},
            {TENANT_METADATA_KEY: {"$in": self.tenant_values()}},
        ]

    def predicate(self, *clauses: Dict[str, Any]) -> Dict[str, Any]:
        """Combine caller clauses (e.g. file id) with the scope clauses."""
        return {"$and": [*clauses, *self.scope_clauses()]}

    def owns(self, tenant: Optional[str], owner: Optional[str]) -> bool:
        """Whether a stored row's recorded tenant/owner falls inside this scope."""
        return owner in self.owners and tenant in self.tenant_values()


def file_clause(file_id: str) -> Dict[str, Any]:
    return {"file_id": {"$eq": file_id}}


def files_clause(file_ids: Sequence[str]) -> Dict[str, Any]:
    return {"file_id": {"$in": list(file_ids)}}


def resolve_scope(request: Request, entity_id: Optional[str] = None) -> ScopeFilter:
    """Resolve the caller's scope, or refuse the request.

    Scope comes from the verified token — never from the query, the body, or the
    returned rows. A caller-supplied ``entity_id`` only widens the owner set
    when the token proves the caller may act for that entity.
    """
    principal: Optional[Principal] = getattr(request.state, "principal", None)

    if principal is None:
        # No signing key configured anywhere: preserve the unauthenticated
        # deployment's single-owner behaviour rather than opening the store.
        owner = entity_id or PUBLIC_OWNER
        return ScopeFilter(tenant=BASE_TENANT_ID, owners=(owner,))

    if principal.tenant == SYSTEM_TENANT_ID:
        # __SYSTEM__ is a query-time wildcard upstream; porting it here would
        # hand every background context cross-tenant reach.
        logger.warning("Refused system-tenant request | subject=%s", principal.subject)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System tenant may not read documents",
        )

    owners = {principal.subject}
    if entity_id:
        if not principal.permits_entity(entity_id):
            logger.warning(
                "Denied entity access | subject=%s entity=%s",
                principal.subject,
                entity_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized for the requested entity",
            )
        owners.add(entity_id)

    return ScopeFilter(tenant=principal.tenant, owners=tuple(sorted(owners)))


def token_scope(request: Request) -> ScopeFilter:
    """Scope for endpoints that take no ``entity_id``: the token decides alone."""
    principal: Optional[Principal] = getattr(request.state, "principal", None)
    if principal is None:
        return ScopeFilter(tenant=BASE_TENANT_ID, owners=(PUBLIC_OWNER,))
    scope = resolve_scope(request)
    if principal.entities:
        return ScopeFilter(
            tenant=scope.tenant,
            owners=tuple(sorted(set(scope.owners) | set(principal.entities))),
        )
    return scope


def writer_tenant(request: Request) -> str:
    """Tenant to stamp on chunks written by this request."""
    principal: Optional[Principal] = getattr(request.state, "principal", None)
    if principal is None or not principal.tenant:
        return BASE_TENANT_ID
    if principal.tenant == SYSTEM_TENANT_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System tenant may not write documents",
        )
    return principal.tenant
