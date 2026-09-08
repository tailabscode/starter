# Engineering Teams

## Aurora Team

The Aurora team owns shared caching infrastructure used across the payments group. Besides the
Nimbus merchant risk cache, Aurora also owns the rate limiter that protects the checkout API and
the session-token cache used by the login service. Aurora was formed in 2023 to consolidate
caching work that used to be duplicated by every team that needed it.

## Fraud Policy Team

The fraud policy team defines the business rules that Nimbus scoring workers evaluate. They do
not write Nimbus code themselves; they submit rule changes through a review process that Nimbus
maintainers implement.

## Platform Reliability Team

The platform reliability team runs on-call for Nimbus, the checkout API, and the login service.
They are not responsible for the correctness of caching logic, only for keeping the services
that depend on it running.
