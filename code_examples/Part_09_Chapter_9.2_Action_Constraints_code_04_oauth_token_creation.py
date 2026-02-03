# Avoid: Monolithic tokens grant everything
oauth_token = create_token(scopes=['full_access'])

# Correct: Fine-grained scopes limit to specific operations
oauth_token = create_token(scopes=[
    'read:calendar',      # Can read calendar events
    'write:calendar',     # Can create calendar events
    # Explicitly excluded: 'read:email', 'send:email', 'admin:*'
])