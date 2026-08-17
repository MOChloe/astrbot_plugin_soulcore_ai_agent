"""Windows owner-only ACL publication for private SoulCore paths."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Any

_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TRUSTEE_W(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),
    ]


class EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", TRUSTEE_W),
    ]


def restrict_windows_owner_only(path: Path, *, inherit_to_children: bool) -> None:
    """Publish a protected DACL granting full control only to the process owner."""

    advapi, kernel = _windows_security_apis()
    token = _open_process_token(advapi, kernel)
    acl = ctypes.c_void_p()
    try:
        sid, _token_info = _process_owner_sid(advapi, token)
        acl = _owner_full_control_acl(advapi, sid, inherit_to_children)
        _publish_owner_acl(advapi, kernel, path, sid, acl)
    finally:
        if acl:
            kernel.LocalFree(acl)
        kernel.CloseHandle(token)


def _windows_security_apis() -> tuple[Any, Any]:
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.SetEntriesInAclW.argtypes = [
        wintypes.ULONG,
        ctypes.POINTER(EXPLICIT_ACCESS_W),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.SetEntriesInAclW.restype = wintypes.DWORD
    advapi.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi.EqualSid.restype = wintypes.BOOL
    return advapi, kernel


def _open_process_token(advapi: Any, kernel: Any) -> wintypes.HANDLE:
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    return token


def _process_owner_sid(advapi: Any, token: wintypes.HANDLE) -> tuple[ctypes.c_void_p, Any]:
    required = wintypes.DWORD()
    advapi.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(required))
    if not required.value:
        raise ctypes.WinError(ctypes.get_last_error())
    token_info = ctypes.create_string_buffer(required.value)
    if not advapi.GetTokenInformation(
        token,
        _TOKEN_USER,
        token_info,
        required,
        ctypes.byref(required),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    sid = ctypes.cast(token_info, ctypes.POINTER(SID_AND_ATTRIBUTES)).contents.Sid
    return sid, token_info


def _owner_full_control_acl(
    advapi: Any, sid: ctypes.c_void_p, inherit_to_children: bool
) -> ctypes.c_void_p:
    access = EXPLICIT_ACCESS_W(
        0x001F01FF,
        2,
        0x3 if inherit_to_children else 0,
        TRUSTEE_W(None, 0, 0, 1, sid),
    )
    acl = ctypes.c_void_p()
    code = advapi.SetEntriesInAclW(1, ctypes.byref(access), None, ctypes.byref(acl))
    if code:
        raise ctypes.WinError(code)
    return acl


def _publish_owner_acl(
    advapi: Any,
    kernel: Any,
    path: Path,
    sid: ctypes.c_void_p,
    acl: ctypes.c_void_p,
) -> None:
    owner_matches = windows_path_owner_matches(advapi, kernel, path, sid)
    security_information = _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION
    code = advapi.SetNamedSecurityInfoW(
        str(path),
        _SE_FILE_OBJECT,
        security_information | (0 if owner_matches else _OWNER_SECURITY_INFORMATION),
        None if owner_matches else sid,
        None,
        acl,
        None,
    )
    if code:
        raise ctypes.WinError(code)
    if not windows_path_owner_matches(advapi, kernel, path, sid):
        raise ctypes.WinError(1307)


def windows_path_owner_matches(
    advapi: Any,
    kernel: Any,
    path: Path,
    expected_sid: ctypes.c_void_p,
) -> bool:
    """Read and compare an object's owner SID without retaining the descriptor."""

    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    code = advapi.GetNamedSecurityInfoW(
        str(path),
        _SE_FILE_OBJECT,
        _OWNER_SECURITY_INFORMATION,
        ctypes.byref(owner),
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    if code:
        raise ctypes.WinError(code)
    try:
        return bool(owner.value and advapi.EqualSid(owner, expected_sid))
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)


__all__ = ["restrict_windows_owner_only"]
