---
title: Code examples
nav_order: 10
permalink: /code-examples/
---

# Code examples
{: .no_toc }

402 standalone snippets under [`code_examples/`]({{ site.repo_tree }}/code_examples), pulled from the chapter text, plus
20 additional worked examples under [`more_examples/`]({{ site.repo_tree }}/more_examples).

> **These files are examples only.** They illustrate concepts discussed in the chapters. Expect to put in extra work to
> mature them into proofs of concept, and do not use them in production as written. If one demonstrates an unsafe
> pattern without saying so, please [report it]({{ site.repo_blob }}/SECURITY.md).

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Naming

Files in `code_examples/` follow `Part_NN_Chapter_X.Y_<topic>_code_NN_<name>.<ext>`, so the chapter is in the name. They
are Python (373), YAML (17), shell (11), and JavaScript (1).
Each Part page lists its files under the chapter they belong to.

## Files per Part

| Part | Snippets in `code_examples/` | Files in `more_examples/` |
|---|---:|---:|
| [Part 1]({{ site.baseurl }}/curriculum/part-01/) | 53 | 4 |
| [Part 2]({{ site.baseurl }}/curriculum/part-02/) | 66 | 1 |
| [Part 3]({{ site.baseurl }}/curriculum/part-03/) | 35 | 2 |
| [Part 4]({{ site.baseurl }}/curriculum/part-04/) | 45 | — |
| [Part 5]({{ site.baseurl }}/curriculum/part-05/) | 8 | 2 |
| [Part 6]({{ site.baseurl }}/curriculum/part-06/) | 59 | 2 |
| [Part 7]({{ site.baseurl }}/curriculum/part-07/) | 76 | 5 |
| [Part 8]({{ site.baseurl }}/curriculum/part-08/) | 29 | 2 |
| [Part 9]({{ site.baseurl }}/curriculum/part-09/) | 27 | — |
| [Part 10]({{ site.baseurl }}/curriculum/part-10/) | 4 | 2 |

Some files are named for chapter numbers that are not in the current chapter list (`3.1`, `4.1C`, `4.2A`, `4.2B`,
`10.3`); the Part pages list them separately as *Other code examples*.

## Additional worked examples

Larger examples for selected Parts (no examples exist for Parts 4 and 9):

| Part | Files |
|---|---|
| Part 1 | [`complex_workflow.py`]({{ site.repo_blob }}/more_examples/part_01/complex_workflow.py), [`distributed_caching.py`]({{ site.repo_blob }}/more_examples/part_01/distributed_caching.py), [`load_balanced_agents.py`]({{ site.repo_blob }}/more_examples/part_01/load_balanced_agents.py), [`state_machine_agent.py`]({{ site.repo_blob }}/more_examples/part_01/state_machine_agent.py) |
| Part 2 | [`streaming_agent.py`]({{ site.repo_blob }}/more_examples/part_02/streaming_agent.py) |
| Part 3 | [`ab_testing.py`]({{ site.repo_blob }}/more_examples/part_03/ab_testing.py), [`evaluation_pipeline.py`]({{ site.repo_blob }}/more_examples/part_03/evaluation_pipeline.py) |
| Part 5 | [`hierarchical_planner.py`]({{ site.repo_blob }}/more_examples/part_05/hierarchical_planner.py), [`replanning_agent.py`]({{ site.repo_blob }}/more_examples/part_05/replanning_agent.py) |
| Part 6 | [`etl_pipeline.py`]({{ site.repo_blob }}/more_examples/part_06/etl_pipeline.py), [`incremental_updates.py`]({{ site.repo_blob }}/more_examples/part_06/incremental_updates.py) |
| Part 7 | [`nim_autoscaler.py`]({{ site.repo_blob }}/more_examples/part_07/nim_autoscaler.py), [`nim_cost_tracker.py`]({{ site.repo_blob }}/more_examples/part_07/nim_cost_tracker.py), [`nim_load_balancer.py`]({{ site.repo_blob }}/more_examples/part_07/nim_load_balancer.py), [`nim_monitoring_stack.yaml`]({{ site.repo_blob }}/more_examples/part_07/nim_monitoring_stack.yaml), [`nim_production_deployment.yaml`]({{ site.repo_blob }}/more_examples/part_07/nim_production_deployment.yaml) |
| Part 8 | [`custom_dashboard.py`]({{ site.repo_blob }}/more_examples/part_08/custom_dashboard.py), [`grafana_dashboard.json`]({{ site.repo_blob }}/more_examples/part_08/grafana_dashboard.json) |
| Part 10 | [`explanation_generator.py`]({{ site.repo_blob }}/more_examples/part_10/explanation_generator.py), [`traceable_agent.py`]({{ site.repo_blob }}/more_examples/part_10/traceable_agent.py) |
