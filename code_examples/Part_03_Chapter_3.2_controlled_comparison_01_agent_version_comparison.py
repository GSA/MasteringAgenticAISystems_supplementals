from scipy import stats
import numpy as np

# Comparing two agent versions with controlled methodology
def compare_agent_versions(test_cases, num_trials=10):
    """
    Run both agents num_trials times on the same test cases, average each case's
    score over its trials, then compare the agents case by case.
    Returns statistical significance assessment.
    """
    control_scores = []    # one row per trial, one column per test case
    treatment_scores = []

    for trial in range(num_trials):
        # Seeds make runs easier to log and rerun, but they don't pair the two agents,
        # and hosted models are not fully deterministic even with a fixed seed
        seed = 42 + trial

        # Evaluate control agent: returns one score per test case (1 = pass, 0 = fail)
        control_case_scores = evaluate_agent(
            agent='control_v1',
            test_cases=test_cases,
            random_seed=seed,
            hyperparameters={'temperature': 0.7, 'max_tokens': 1024}
        )
        control_scores.append(control_case_scores)

        # Evaluate treatment agent with identical conditions
        treatment_case_scores = evaluate_agent(
            agent='enhanced_v2',
            test_cases=test_cases,
            random_seed=seed,
            hyperparameters={'temperature': 0.7, 'max_tokens': 1024}
        )
        treatment_scores.append(treatment_case_scores)

    # Average the repeated trials within each test case. The test cases, not the
    # trials, are the independent units, so the comparison is paired by case.
    control_per_case = np.mean(control_scores, axis=0)
    treatment_per_case = np.mean(treatment_scores, axis=0)
    control_mean = np.mean(control_per_case)
    treatment_mean = np.mean(treatment_per_case)
    improvement = treatment_mean - control_mean

    # Paired t-test across test cases
    t_statistic, p_value = stats.ttest_rel(treatment_per_case, control_per_case)

    # Calculate confidence interval for improvement
    differences = treatment_per_case - control_per_case
    ci_95 = stats.t.interval(0.95, len(differences)-1,
                             loc=np.mean(differences),
                             scale=stats.sem(differences))

    # Run-to-run spread: standard deviation of each trial's overall accuracy
    print(f"Control: {control_mean:.1%} ± {np.std(np.mean(control_scores, axis=1)):.1%}")
    print(f"Treatment: {treatment_mean:.1%} ± {np.std(np.mean(treatment_scores, axis=1)):.1%}")
    print(f"Improvement: {improvement:.1%} (95% CI: [{ci_95[0]:.1%}, {ci_95[1]:.1%}])")
    print(f"Statistical significance: p={p_value:.4f}")

    # Apply decision threshold: require both statistical significance and practical significance
    if p_value < 0.05 and improvement > 0.03:  # 3 percentage points minimum practical improvement
        return "DEPLOY", improvement, p_value
    elif p_value < 0.05 and improvement > 0:
        return "MARGINAL", improvement, p_value
    else:
        return "KEEP_CONTROL", improvement, p_value

# Example usage: report each benchmark separately rather than pooling them
for benchmark in ['agentbench', 'webarena', 'hotpotqa']:
    decision, delta, significance = compare_agent_versions(
        test_cases=load_benchmark_suite([benchmark]),
        num_trials=10
    )
