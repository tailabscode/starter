# Nimbus Deployment

## Environments

Nimbus runs in three environments: dev, staging, and production. Only production reads real
transaction traffic; dev and staging replay a sampled, anonymized feed for testing.

## Rollout Process

Deploys to production use a canary rollout: a new version first takes 5 percent of traffic for
15 minutes, and is promoted to 100 percent only if error rates and scoring latency stay within
bounds. A failed canary automatically rolls back without paging anyone.

## On-Call

The Nimbus on-call rotation is staffed by the platform reliability team, not by the Aurora team.
Aurora is only paged for incidents that are specifically traced back to the shared cache.
