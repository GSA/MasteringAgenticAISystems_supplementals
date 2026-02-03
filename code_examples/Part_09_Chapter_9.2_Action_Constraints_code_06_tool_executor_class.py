class ToolExecutor:
    """Enforce constraints at tool level with domain-specific logic"""

    HIGH_RISK_TOOLS = ['file_delete', 'db_modify', 'shell_exec']

    def execute_tool(self, tool_name: str, params: dict, context: dict) -> Any:
        """Execute tool with safety checks appropriate to risk level"""
        # High-risk tools require explicit approval flag
        if tool_name in self.HIGH_RISK_TOOLS:
            if not context.get('high_risk_approved', False):
                raise PermissionError(
                    f"Tool '{tool_name}' requires explicit approval"
                )

        # Validate parameters with tool-specific logic
        if tool_name == 'file_delete':
            file_path = params.get('path')
            if not self._is_safe_path(file_path):
                raise ValueError(
                    f"Cannot delete {file_path} - outside safe zone"
                )

        # Execute with monitoring
        return self._execute(tool_name, params)

    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is within allowed directories"""
        safe_dirs = ['/tmp/agent_workspace', '/data/staging']
        return any(path.startswith(safe_dir) for safe_dir in safe_dirs)