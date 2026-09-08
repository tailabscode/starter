# Nimbus Architecture

## Components

Nimbus has three main components: the ingest gateway, the scoring workers, and the alert
publisher. Events arrive at the ingest gateway over Kafka, get enriched and scored by the
workers, and any event that crosses a risk threshold is pushed to the alert publisher.

## Ingest Gateway

The ingest gateway validates event schemas and rejects malformed events before they reach the
scoring workers. It is stateless and can be scaled horizontally without coordination.

## Scoring Workers

Scoring workers apply fraud rules to each event using the merchant risk cache described in the
platform overview. Workers are the only component that reads from the shared cache; they never
write to it directly.

## Alert Publisher

The alert publisher batches high-risk events and forwards them to the case-management system
used by human fraud analysts. It applies simple deduplication so the same merchant does not
generate duplicate alerts within a five-minute window.
