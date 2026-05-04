"""Entry point for multicamera stop-reason rule-aware experiments.

This intentionally reuses the core rule-aware trainer. The experiment is
separated at the command level so front-only and multicamera pseudo-label runs
can be tracked with different scripts, output folders, and logs.
"""

from .train_ruleaware import main


if __name__ == "__main__":
    main()
