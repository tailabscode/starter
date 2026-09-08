# Nimbus Roadmap

## Next Quarter

The Nimbus team plans to add a secondary scoring pass for cross-border transactions, which
currently use the same rules as domestic ones despite having a different risk profile.

## Under Discussion

There is an open proposal to let the alert publisher call out to a third-party device-fingerprint
service before forwarding an alert, which would reduce false positives but add external latency.
No decision has been made and it is not currently funded.

## Explicitly Out of Scope

Real-time model retraining is explicitly out of scope for Nimbus. Model updates are still
deployed as a manual, reviewed process, and there are no plans to automate that pipeline.
