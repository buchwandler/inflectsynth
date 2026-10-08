---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.1
kind: added
summary: Added prepared-request token measurement and optional capacity enforcement
status: accepted
audience: null
scopes: []
source_refs:
  - git:17aa803e650faf85aed6b5b200da9392cedf844f
paths:
  - README.md
  - benchmarks/data/request_capacity_policy.json
  - benchmarks/data/request_capacity_stimuli.json
  - benchmarks/promote_request_capacity.py
  - benchmarks/request_capacity_benchmark.py
  - examples/all_models.py
  - examples/basic.py
  - examples/calibrated.py
  - inflectsynth/__init__.py
  - inflectsynth/__main__.py
  - inflectsynth/api_contract.py
  - inflectsynth/constants.py
  - inflectsynth/errors.py
  - inflectsynth/types.py
  - inflectsynth/voice.py
  - inflectsynth/voice_level.py
  - tests/test_cli.py
  - tests/test_promote_request_capacity.py
  - tests/test_public_api.py
  - tests/test_request_api_contract.py
  - tests/test_request_capacity_benchmark.py
  - tests/test_request_measurement.py
  - tests/test_voice_contract.py
issues: []
prs: []
sources:
  - git:17aa803e650faf85aed6b5b200da9392cedf844f
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
