from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

def analyze_feedback_themes(feedback_texts, n_themes=5):
    """
    Extract recurring themes from user feedback using embedding-based clustering.
    Returns theme clusters with representative examples and frequency counts.
    """
    # Generate embeddings for semantic clustering
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(feedback_texts)

    # Cluster feedback into themes
    kmeans = KMeans(n_clusters=n_themes, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Analyze each theme
    themes = []
    for theme_id in range(n_themes):
        # Get feedback in this cluster
        theme_feedback = [feedback_texts[i] for i in range(len(feedback_texts))
                         if clusters[i] == theme_id]

        # Find representative example (closest to cluster center)
        theme_embeddings = embeddings[clusters == theme_id]
        center_dist = np.linalg.norm(theme_embeddings - kmeans.cluster_centers_[theme_id], axis=1)
        representative_idx = np.argmin(center_dist)

        themes.append({
            'theme_id': theme_id,
            'frequency': len(theme_feedback),
            'percentage': len(theme_feedback) / len(feedback_texts) * 100,
            'representative_example': theme_feedback[representative_idx],
            'all_examples': theme_feedback[:10]  # Store top 10 for review
        })

    # Sort by frequency
    themes.sort(key=lambda x: x['frequency'], reverse=True)

    return themes

def prioritize_improvements(themes, implicit_signals):
    """
    Combine feedback themes with behavioral signals to prioritize improvements.
    Returns ranked improvement opportunities with impact estimates.
    """
    priorities = []

    for theme in themes:
        # Calculate impact score combining frequency, severity, and implicit signals
        frequency_score = theme['percentage'] / 100  # Normalize to 0-1

        # Estimate severity from sentiment (simplified)
        severity_score = estimate_severity(theme['representative_example'])

        # Correlate with implicit signals
        affected_users = estimate_affected_users(theme, implicit_signals)

        # Combined priority score
        impact_score = (frequency_score * 0.4 +
                       severity_score * 0.3 +
                       affected_users * 0.3)

        priorities.append({
            'theme': theme['representative_example'],
            'frequency': theme['frequency'],
            'severity': severity_score,
            'affected_users': affected_users,
            'impact_score': impact_score,
            'recommended_action': suggest_action(theme)
        })

    # Sort by impact score
    priorities.sort(key=lambda x: x['impact_score'], reverse=True)

    return priorities

# Example usage
feedback_data = load_production_feedback(days=30)  # Last 30 days
implicit_data = load_behavioral_signals(days=30)

# Extract themes
themes = analyze_feedback_themes(
    [f['comment'] for f in feedback_data if f['comment']],
    n_themes=8
)

# Prioritize improvements
improvement_priorities = prioritize_improvements(themes, implicit_data)

# Display top priorities
for i, priority in enumerate(improvement_priorities[:5], 1):
    print(f"\n{i}. {priority['theme']}")
    print(f"   Impact: {priority['impact_score']:.2f}")
    print(f"   Frequency: {priority['frequency']} reports ({priority['frequency']/len(feedback_data)*100:.1f}%)")
    print(f"   Affected users: ~{priority['affected_users']:.0f}")
    print(f"   Recommended: {priority['recommended_action']}")
