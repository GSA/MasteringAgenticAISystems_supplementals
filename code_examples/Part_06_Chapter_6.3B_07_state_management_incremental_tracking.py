    def _get_last_run_time(self) -> datetime:
        """
        Get timestamp of last successful ETL run.

        Enables incremental extraction by tracking processing windows.
        Defaults to configured lookback period if no state exists.

        Returns:
            datetime: Timestamp to use for incremental extraction
        """
        state_file = Path(self.config["incremental"]["state_file"])

        if state_file.exists():
            state = json.loads(state_file.read_text())
            return datetime.fromisoformat(state["last_run"])

        # Default: look back configured hours for first run
        return datetime.now() - timedelta(
            hours=self.config["incremental"]["lookback_hours"]
        )

    def _save_run_state(self, timestamp: datetime):
        """
        Save ETL run state for incremental tracking.

        Records timestamp of successful completion to enable
        next run to process only subsequent changes.

        Args:
            timestamp (datetime): Completion timestamp to record
        """
        state_file = Path(self.config["incremental"]["state_file"])

        state = {
            "last_run": timestamp.isoformat(),
            "status": "success"
        }

        state_file.write_text(json.dumps(state, indent=2))
