---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.2
kind: added
summary:
  Added validated 400-token request budgets for managed Nano v2 and Micro v2
  models
status: accepted
audience: null
scopes: []
source_refs:
  - git:a61b507ae2b7bd796dcda9ff42350ca9a174693a
paths:
  - README.md
  - benchmarks/promote_request_capacity.py
  - docs/changelog.md
  - inflectsynth/__init__.py
  - inflectsynth/api_contract.py
  - inflectsynth/data/request_capacity.json
  - inflectsynth/discovery.py
  - inflectsynth/request_capacity.py
  - inflectsynth/types.py
  - inflectsynth/voice.py
  - tests/test_discovery.py
  - tests/test_promote_request_capacity.py
  - tests/test_request_api_contract.py
  - tests/test_request_capacity_policy.py
  - tests/test_request_measurement.py
issues: []
prs: []
sources:
  - git:a61b507ae2b7bd796dcda9ff42350ca9a174693a
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
