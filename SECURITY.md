# Security Policy

This repository is primarily study material: Markdown labs, slides, quizzes,
and videos, plus roughly 400 illustrative Python/YAML snippets under
`code_examples/` (and similar snippets embedded in `labs/`). There is no
deployed service, installable package, or running application associated
with this repository, so "security" here is narrower than for a typical
software project. It generally falls into two categories:

1. **An insecure pattern in an illustrative code snippet** — for example, a
   `code_examples/` or `labs/` snippet that demonstrates or implies an
   unsafe practice (hardcoded secrets, injection-prone code, disabled
   certificate validation, etc.) without a warning that it is deliberately
   simplified or that the pattern should not be used as-is in production.
2. **A supply-chain concern in the repository's own tooling** — notably,
   the pre-commit hook at
   [`.github/hooks/pre-commit`](.github/hooks/pre-commit), which on every
   commit fetches and executes a script from the
   [`GSA/odp-code-repository-commit-rules`](https://github.com/GSA/odp-code-repository-commit-rules)
   repository (a gitleaks-based secret-scanning hook). If you have a concern
   about this fetch-and-execute pattern, the pinning/versioning of that
   script, or its behavior, please report it using the process below.

## Reporting a Vulnerability

**Preferred channel:** Use GitHub's private vulnerability reporting feature
for this repository: go to the **Security** tab and select
**Report a vulnerability**. This creates a private advisory that only
maintainers and GitHub can see, which keeps any sensitive details out of
public issues until a fix or a determination is made.

**Fallback channel:** If you cannot use GitHub Security Advisories (for
example, you don't have a GitHub account), email
[cto@gsa.gov](mailto:cto@gsa.gov) with as much detail as you can provide:
the affected file(s), the nature of the concern, and any suggested fix.

Please do **not** open a public GitHub issue for a security report until a
maintainer has confirmed it's appropriate to do so.

## What to include

* File path(s) and, where relevant, line numbers or a link to the exact
  snippet.
* Why the pattern is a concern (what it demonstrates and what the risk is).
* A suggested fix or mitigation, if you have one (e.g., adding a warning
  comment, or correcting the snippet).

## Response times

We aim to acknowledge security reports within **3 business days**. Given
this is a documentation/example repository maintained by a small,
part-time team, remediation timelines will vary depending on the finding —
a missing warning comment can be fixed quickly, while a broader review of
snippets in a given lab may take longer.

## Scope notes

* There is no bug-bounty program associated with this repository.
* This policy does not cover vulnerabilities in third-party tools,
  libraries, or platforms referenced by the material (e.g., a CVE in a
  library used by an example) — please report those upstream to the
  relevant project. If the reference in this repository should be updated
  as a result (e.g., pin a version, add a note), feel free to also flag that
  here.
