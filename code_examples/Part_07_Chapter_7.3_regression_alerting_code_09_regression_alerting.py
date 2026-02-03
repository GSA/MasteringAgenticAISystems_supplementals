# After prompt modification deployment
results = evaluator.evaluate()

# Automated alert triggered
if results["overall_metrics"]["accuracy"] < evaluator.accuracy_threshold:
    alert_team(
        severity="HIGH",
        message=f"Agent accuracy degraded to {results['overall_metrics']['accuracy']:.2%}, below {evaluator.accuracy_threshold:.2%} threshold",
        affected_tests=results["regressions_detected"][0]["affected_tests"],
        previous_run="2024-11-10-14:32:15",
        current_run="2024-11-10-16:45:30"
    )
