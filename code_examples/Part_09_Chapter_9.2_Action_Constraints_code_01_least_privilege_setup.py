# Avoid: Cloning admin profiles grants all permissions
agent_permissions = copy(admin_profile)  # Every permission in the system

# Correct: Surgical precision grants exactly what's needed
agent_permissions = PermissionSet(
    objects=['Account', 'Contact'],           # Only these two object types
    operations=['read', 'create'],            # Not 'update' or 'delete'
    fields=['name', 'email', 'phone'],        # Exclude sensitive fields
    exclude=['delete', 'modify_permissions']  # Explicitly block dangerous ops
)