# Nimbus Incident History

## 2023-11 Inconsistent Scores Incident

Before the 2024 cache rewrite, a deploy caused half the scoring workers to hold stale merchant
risk data while the other half had fresh data. The same merchant received different risk scores
depending on which worker handled the event, and several legitimate transactions were declined.
The root cause was the old per-node cache design; it was the main motivation for the rewrite
described in the platform overview.

## 2024-09 Cache Latency Spike

After the rewrite, a misconfigured connection pool on the shared cache caused a brief latency
spike during a traffic peak. The Aurora team resolved it by increasing the pool size; scoring
latency was unaffected for more than a few seconds.

## 2025-02 Alert Duplication Bug

A bug in the alert publisher's deduplication window caused a small number of merchants to
receive duplicate alerts. This was unrelated to caching and was fixed by the Nimbus maintainers
directly.
