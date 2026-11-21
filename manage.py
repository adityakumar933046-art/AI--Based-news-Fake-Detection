
#!/usr/bin/env python
"""Django command-line utility."""

import os
import sys


def main():
    """Run Django management commands."""
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "fake_news.settings"
    )

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django could not be imported. "
            "Check whether Django is installed "
            "in your active Python environment."
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
