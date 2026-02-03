// Format evaluation results as PR comment
function formatComment(results) {
  if (results.error) {
    return `## ⚠️ Evaluation Failed

**Error:** ${results.error}

The evaluation pipeline encountered an error and could not complete. Please check the workflow logs for details.`;
  }

  const passed = results.passed;
  const icon = passed ? '✅' : '❌';
  const status = passed ? 'PASSED' : 'FAILED';

  const metrics = results.metrics;
  const baseline = results.baseline;
  const comparison = results.comparison;

  // Build comparison table
  const comparisonTable = `
| Metric | Baseline | Current | Change | Status |
|--------|----------|---------|--------|--------|
| Accuracy | ${(baseline.accuracy * 100).toFixed(1)}% | ${(metrics.accuracy * 100).toFixed(1)}% | ${comparison.accuracy_change > 0 ? '+' : ''}${(comparison.accuracy_change * 100).toFixed(1)}% | ${comparison.accuracy_passed ? '✓' : '⚠️'} |
| P95 Latency | ${baseline.latency_p95.toFixed(0)}ms | ${metrics.latency_p95.toFixed(0)}ms | ${comparison.latency_change > 0 ? '+' : ''}${(comparison.latency_change * 100).toFixed(1)}% | ${comparison.latency_passed ? '✓' : '⚠️'} |
| Empathy Score | ${baseline.empathy_mean.toFixed(2)} | ${metrics.empathy_mean.toFixed(2)} | ${comparison.empathy_change > 0 ? '+' : ''}${comparison.empathy_change.toFixed(2)} | ${comparison.empathy_passed ? '✓' : '⚠️'} |
| Compliance Rate | ${(baseline.compliance_rate * 100).toFixed(1)}% | ${(metrics.compliance_rate * 100).toFixed(1)}% | ${comparison.compliance_change > 0 ? '+' : ''}${(comparison.compliance_change * 100).toFixed(1)}% | ${comparison.compliance_passed ? '✓' : '⚠️'} |
`;

  return `## ${icon} Agent Evaluation: ${status}

${comparisonTable}

### Summary
- **Test Dataset:** ${metrics.test_count} queries
- **Overall Status:** ${passed ? 'All quality gates passed' : 'Some metrics regressed beyond thresholds'}

${!passed ? '### ⚠️ Regressions Detected\n' + comparison.alerts.map(a => `- ${a}`).join('\n') : ''}

<details>
<summary>View detailed metrics</summary>

\`\`\`json
${JSON.stringify(metrics, null, 2)}
\`\`\`

</details>`;
}

// Post comment
const comment = formatComment(results);
await github.rest.issues.createComment({
  owner: context.repo.owner,
  repo: context.repo.repo,
  issue_number: context.issue.number,
  body: comment
});
