# Code examples

These files illustrate concepts from the study plan. They are **not** one cohesive application and they do not share a single dependency environment.

## Example types

- **Standalone examples** should parse in their declared language and are checked by `make validate`.
- **Contextual excerpts** are methods or continuation snippets intended to be combined with adjacent chapter examples. They are listed in [`fragments.txt`](fragments.txt), parse as valid Python modules, and are not expected to run independently.
- **Exercise scaffolds** intentionally retain `TODO`, `TBD`, or similar implementation markers for learner work. They are listed in [`scaffolds.txt`](scaffolds.txt) and are not complete solutions.
- **Infrastructure examples** may require cloud accounts, GPUs, Kubernetes, licensed services, or provider-specific configuration.

## Before running an example

1. Read the corresponding chapter and neighboring example files.
2. Create an isolated virtual environment or container.
3. Select and pin dependency versions appropriate to the API style shown.
4. Replace sample endpoints and environment variables; never commit credentials.
5. Validate behavior, security, cost, accessibility, and performance in your own environment before production use.

Several examples use APIs that have since changed. Review [`COMPATIBILITY.md`](COMPATIBILITY.md) for known migrations and provider documentation.
