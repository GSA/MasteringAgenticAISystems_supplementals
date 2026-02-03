from scipy import stats
import numpy as np

# Comparing two agent versions with controlled methodology
def compare_agent_versions(test_cases, num_trials=10):
    """
    Run controlled comparison across multiple trials with different random seeds.
    Returns statistical significance assessment.
    """
    control_results = []
    treatment_results = []

    for trial in range(num_trials):
        # Use different random seed per trial but ensure both agents use same seed
        seed = 42 + trial

        # Evaluate control agent
        control_accuracy = evaluate_agent(
            agent='control_v1',
            test_cases=test_cases,
            random_seed=seed,
            hyperparameters={'temperature': 0.7, 'max_tokens': 1024}
        )
        control_results.append(control_accuracy)

        # Evaluate treatment agent with identical conditions
        treatment_accuracy = evaluate_agent(
            agent='enhanced_v2',
            test_cases=test_cases,
            random_seed=seed,
            hyperparameters={'temperature': 0.7, 'max_tokens': 1024}
        )
        treatment_results.append(treatment_accuracy)

    # Calculate statistics
    control_mean = np.mean(control_results)
    treatment_mean = np.mean(treatment_results)
    improvement = treatment_mean - control_mean

    # Paired t-test (samples are paired by random seed)
    t_statistic, p_value = stats.ttest_rel(treatment_results, control_results)

    # Calculate confidence interval for improvement
    differences = np.array(treatment_results) - np.array(control_results)
    ci_95 = stats.t.interval(0.95, len(differences)-1,
                             loc=np.mean(differences),
                             scale=stats.sem(differences))

    print(f"Control: {control_mean:.1%} ± {np.std(control_results):.1%}")
    print(f"Treatment: {treatment_mean:.1%} ± {np.std(treatment_results):.1%}")
    print(f"Improvement: {improvement:.1%} (95% CI: [{ci_95[0]:.1%}, {ci_95[1]:.1%}])")
    print(f"Statistical significance: p={p_value:.4f}")

    # Apply decision threshold: require both statistical significance and practical significance
    if p_value < 0.05 and improvement > 0.03:  # 3% minimum practical improvement
        return "DEPLOY", improvement, p_value
    elif p_value < 0.05 and improvement > 0:
        return "MARGINAL", improvement, p_value
    else:
        return "KEEP_CONTROL", improvement, p_value

# Example usage
decision, delta, significance = compare_agent_versions(
    test_cases=load_benchmark_suite(['agentbench', 'webarena', 'hotpotqa']),
    num_trials=10
)
