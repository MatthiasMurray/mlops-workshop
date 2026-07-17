# Open Work

This repo should remain the workshop application/runtime repo, not the source of truth for AWS account governance.

## Current Operating Posture

- The workshop runtime is intentionally parked to avoid standing cost.
- `SERVICE_DESIRED_COUNT=0` in `deploy-vars.txt` keeps services scaled down after pipeline runs.
- To teach again, set `SERVICE_DESIRED_COUNT=1` and trigger the intended pipeline.

## Repo Boundary

This repo should own:

- Workshop service code
- Container build and deploy behavior
- Participant onboarding for workshop application flows
- Park/resume behavior for workshop runtime

This repo should not own:

- AWS Organizations account lifecycle
- IAM Identity Center governance
- Central budgets and org-level guardrails

Those belong in `aws-org-mgt`.

## Near-Term Tickets

1. Validate park/resume behavior through CI/CD
   - GitHub issue: `mlops-workshop#57`
   - Confirm `SERVICE_DESIRED_COUNT=0` survives pipeline runs
   - Confirm `SERVICE_DESIRED_COUNT=1` cleanly restores runtime for teaching

2. Audit stale workshop runtime artifacts
   - GitHub issue: `mlops-workshop#58`
   - Inventory ALB listener rules, target groups, pipelines, build projects, ECR repos, and secrets
   - Remove or document stale artifacts without breaking the teachable path

3. Clarify future teaching workflow
   - GitHub issue: `mlops-workshop#59`
   - Document the exact sequence to prepare, run, and park a workshop session
   - Keep workshop-specific instructions separate from org/account governance