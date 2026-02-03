import json
import sys

# Load results and check passed flag
with open('results.json', 'r') as f:
    results = json.load(f)

if not results.get('passed', False):
    print('❌ Quality gates failed')
    print('Evaluation metrics regressed beyond acceptable thresholds.')
    print('See PR comment for details.')
    sys.exit(1)
else:
    print('✅ Quality gates passed')
    sys.exit(0)
