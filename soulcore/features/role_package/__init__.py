"""Portable SoulCore role-package format and atomic persistence."""

from .domain import (
    ROLE_PACKAGE_EXTENSION,
    ROLE_PACKAGE_FORMAT_VERSION,
    RolePackageError,
)
from .ports import RolePackageRepositoryPort

__all__ = [
    "ROLE_PACKAGE_EXTENSION",
    "ROLE_PACKAGE_FORMAT_VERSION",
    "RolePackageError",
    "RolePackageRepositoryPort",
]
