"""Configuration management for Article TTS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class Config:
    """Configuration manager for Article TTS."""

    def __init__(self, config_dir: str | Path | None = None):
        """Initialize configuration manager.

        Args:
            config_dir: Configuration directory (default: ~/.article-tts)
        """
        if config_dir is None:
            self.config_dir = Path.home() / '.article-tts'
        else:
            self.config_dir = Path(config_dir)

        self.config_file = self.config_dir / 'config.json'
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
                self._config = {}
        else:
            self._config = {}

    def _save(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value
        self._save()

    def update(self, values: dict[str, Any]) -> None:
        """Update multiple configuration values.

        Args:
            values: Dictionary of key-value pairs to update
        """
        self._config.update(values)
        self._save()

    def delete(self, key: str) -> None:
        """Delete a configuration value.

        Args:
            key: Configuration key to delete
        """
        if key in self._config:
            del self._config[key]
            self._save()

    def clear(self) -> None:
        """Clear all configuration."""
        self._config = {}
        self._save()

    def is_configured(self) -> bool:
        """Check if the tool is configured.

        Returns:
            True if endpoint_id is configured
        """
        return 'endpoint_id' in self._config

    def get_endpoint_id(self) -> str | None:
        """Get the configured endpoint ID.

        Returns:
            Endpoint ID or None if not configured
        """
        return self._config.get('endpoint_id')

    def get_api_key(self) -> str | None:
        """Get the API key (from config or environment).

        Returns:
            API key or None if not found
        """
        # Check config first, then environment
        return self._config.get('api_key') or os.getenv('RUNPOD_API_KEY')

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()

    def display(self) -> None:
        """Display current configuration."""
        if not self._config:
            console.print("[yellow]No configuration found. Run 'article-tts configure' to set up.[/yellow]")
            return

        console.print("\n[bold cyan]Current Configuration:[/bold cyan]")
        console.print(f"  Config file: {self.config_file}")
        console.print()

        for key, value in self._config.items():
            # Mask sensitive values
            if 'key' in key.lower() or 'token' in key.lower():
                display_value = value[:8] + '...' if len(value) > 8 else '***'
            else:
                display_value = value

            console.print(f"  • {key}: {display_value}")
