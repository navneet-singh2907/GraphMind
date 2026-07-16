# HelioDesk reference architecture

HelioDesk is a fictional customer-support workspace operated by Asteria Systems. It combines ticket search, guided answers, workflow automation, and auditable access controls. The production design uses four application services and three shared data systems.

## Request path

1. The **Gateway** authenticates the user, applies the tenant boundary, assigns a request ID, and enforces rate limits.
2. The **Retrieval Engine** receives the scoped question from the Gateway. It searches the Knowledge Index and returns evidence with source references.
3. The **Workflow Worker** executes approved actions such as drafting a reply or routing a ticket. It never publishes a reply without an agent confirmation.
4. The **Audit Store** records authentication, retrieval, and workflow events. Audit records are append-only and retained for 400 days on the Enterprise plan.

The Gateway calls the Retrieval Engine synchronously. The Gateway and Workflow Worker publish audit events asynchronously through the Event Bus. The Audit Store consumes those events and writes them to PostgreSQL. If the Event Bus is unavailable, producers buffer events for up to 15 minutes and retry with the same event ID.

## Data systems

- **Knowledge Index:** Chroma stores document chunks and embeddings for semantic retrieval. Each chunk carries a tenant ID, collection, source URI, and content hash.
- **Relationship Graph:** Neo4j stores normalized entities and relationships extracted from the same chunks. It is an optional retrieval layer, not the system of record.
- **Operational Store:** PostgreSQL stores tenant configuration, audit records, workflow state, and source manifests.
- **Event Bus:** Redis Streams transports workflow and audit events.

## Deployment and controls

Production runs in `us-east` and `eu-west`. Enterprise tenants may pin content and audit data to `eu-west`; application telemetry remains aggregated and contains no document text. Every retrieval request is filtered by tenant before ranking. SAML SSO is available on Enterprise, while all plans support email-based sign-in and role-based access.

The service objective for customer-facing search is 99.95 percent monthly availability. The normal p95 answer latency target is 2.5 seconds. The Reliability Review records the measured Q2 p95 as 2.1 seconds after the April remediation.

## Failure behavior

When Neo4j is unavailable, HelioDesk continues with semantic and keyword retrieval and marks graph evidence as unavailable. When Chroma is unavailable, exact keyword retrieval remains available from the canonical chunks. When the Audit Store is unavailable, actions requiring an audit record are paused, but read-only search continues for up to 15 minutes while events are buffered.
