# Nimbus Platform Overview

Nimbus is the internal event-processing platform used by the payments group to turn raw
transaction events into fraud signals in near real time. It ingests roughly 40,000 events per
second at peak and was originally built as a single monolithic consumer in 2021.

## Why Nimbus Exists

Before Nimbus, fraud rules ran as nightly batch jobs against a data warehouse, which meant a
stolen card could be used for up to 18 hours before anyone noticed. Nimbus moved that scoring
into the event stream itself so a rule can fire within two seconds of a transaction.

## Caching Layer History

Nimbus originally cached merchant risk profiles in a local in-memory map on each consumer node,
which caused inconsistent scores across nodes during deploys. The caching layer was rebuilt from
scratch in 2024 by the Aurora team, who replaced the per-node cache with a shared, versioned
cache that every consumer reads from. That rewrite is the reason merchant risk scores are now
consistent across the whole fleet.

## Current Status

Nimbus is considered stable and is the system of record for real-time fraud scores. Any change
to its scoring logic requires sign-off from the fraud policy team, not just the Nimbus
maintainers.
