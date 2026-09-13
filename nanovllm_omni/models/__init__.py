"""Model families are discovered through the configuration registries.

Model modules stay lazy; import a family module directly when its registry
entry resolves it. This package intentionally exposes no family loader API.
"""

__all__: list[str] = []
