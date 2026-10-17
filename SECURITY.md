# Security policy

This is a personal learning and portfolio project. Nothing in it is deployed,
and it holds no real customer data: every person, organisation and record is
synthetic or fictional.

## Supported versions

Only the `main` branch is maintained.

## Report a vulnerability

Please report privately through GitHub: open the **Security** tab of this
repository and choose **Report a vulnerability** (private vulnerability
reporting is switched on). Please do not open a public issue for a security
problem. Useful details: the file and line or the command that shows the
problem, what an attacker could do with it, and a suggested fix if you have
one. You should get a first reply within a week.

## In scope

- Orchestration patterns, tools, LLM interface and eval code
- Workflows under `.github/`

## Controls in place

- Secret scanning with push protection, Dependabot alerts and security updates
- CodeQL (`security-and-quality`) on every push and weekly
- gitleaks over the full git history in CI
- Every GitHub Action pinned to a commit SHA; workflow tokens are read-only
- A `main` ruleset: no force-push or deletion, CI required on pull requests
