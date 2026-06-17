# Tests import the installed ``agent_team`` package (CI runs them via
# ``uvx --with-editable . --with pytest pytest``; locally use
# ``uv run --with pytest pytest``). No ``sys.path`` shimming is needed —
# the package is importable from its installed/editable location.
