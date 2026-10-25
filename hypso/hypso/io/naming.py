"""Product file names (REFACTOR_PROGRESS.md x42).

Current style, NASA OB.DAAC-like - satellite first, then time, so a plain
listing groups by satellite and sorts by time:

    {SAT}_HSI.{YYYYMMDDTHHMMSS}.{target}.{LEVEL}[.{ac}][.{label}].nc

    HYPSO2_HSI.20250612T095802.aeronetvenice.L1A.nc
    HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.nc
    HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.polymer.nc
    HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.adjusted.nc   (label: experiments)

`.` separates fields and never appears inside one. L2A always carries the
AC field; the optional free-text label is always last. Calibration set and
masking state are not part of the name - they are recorded in the file.
Times are UTC.

Legacy names (`{target}_{YYYY-MM-DDTHH-MM-SSZ}[-{label}]-{level}[-{ac}].nc`,
where label mixed calibration set and masking, e.g. "moved_unmasked") are
still parsed, so existing datasets stay readable. Capture directories keep
the capture ID form `{target}_{YYYY-MM-DDTHH-MM-SSZ}`.
"""
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_LEVEL = re.compile(r"L\d[A-Z]")
_PLATFORM = re.compile(r"([A-Z]+)(\d+)_HSI")
_TIME_FORMAT = "%Y%m%dT%H%M%S"
_CAPTURE_ID_TIME_FORMAT = "%Y-%m-%dT%H-%M-%SZ"

_LEGACY = re.compile(
    r"""
    (?P<target>.+?)_
    (?P<time>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)
    -
    (?:(?P<label>[^-]+)-)?
    (?P<level>l\d[a-z])
    (?:-(?P<ac>[^.]+))?
    \.(?P<ext>\w+)
    """,
    re.VERBOSE,
)


def _check_token(kind: str, value: str) -> str:
    if not _TOKEN.fullmatch(value):
        raise ValueError(f"{kind} {value!r} may only contain letters, digits, '_' and '-'")
    return value


def platform_token(sat_id: str) -> str:
    """"HYPSO-2" -> "HYPSO2_HSI"."""
    token = sat_id.replace("-", "") + "_HSI"
    if not _PLATFORM.fullmatch(token):
        raise ValueError(f"unsupported sat_id {sat_id!r}")
    return token


def sat_id_from_token(token: str) -> str:
    """"HYPSO2_HSI" -> "HYPSO-2"."""
    match = _PLATFORM.fullmatch(token)
    if not match:
        raise ValueError(f"not a platform token: {token!r}")
    return f"{match.group(1)}-{match.group(2)}"


@dataclass(frozen=True)
class ProductName:
    sat_id: Optional[str]           # "HYPSO-2"; None for legacy names (not encoded)
    time: datetime                  # capture start, UTC
    target: str
    level: str                      # "L1A" ... "L1D", "L2A"
    ac: Optional[str] = None        # L2A only, e.g. "polymer", "acolite_l2r"
    label: Optional[str] = None     # experiments only
    legacy_label: Optional[str] = None  # legacy names: e.g. "moved_unmasked"
    extension: str = "nc"
    style: str = "current"          # "current" | "legacy"

    @property
    def capture_id(self) -> str:
        """`{target}_{YYYY-MM-DDTHH-MM-SSZ}` - capture directory name."""
        return f"{self.target}_{self.time.strftime(_CAPTURE_ID_TIME_FORMAT)}"

    def filename(self) -> str:
        """Current-style file name. Requires sat_id (a legacy name doesn't
        carry one - use replace(name, sat_id=...))."""
        if self.sat_id is None:
            raise ValueError("sat_id is required for a current-style name")
        if not _LEVEL.fullmatch(self.level):
            raise ValueError(f"invalid level {self.level!r}")
        if (self.level == "L2A") != (self.ac is not None):
            raise ValueError("the AC field is required for L2A and only allowed there")
        fields = [platform_token(self.sat_id), self.time.strftime(_TIME_FORMAT),
                  _check_token("target", self.target), self.level]
        if self.ac is not None:
            fields.append(_check_token("ac", self.ac))
        if self.label is not None:
            fields.append(_check_token("label", self.label))
        return ".".join(fields) + "." + self.extension

    def with_level(self, level: str, ac: Optional[str] = None) -> "ProductName":
        """Same capture (and label), another level."""
        return replace(self, level=level, ac=ac, legacy_label=None, style="current")


def product_name(sat_id: str, time: datetime, target: str, level: str,
                 ac: str = None, label: str = None) -> ProductName:
    if time.tzinfo is not None:
        time = time.astimezone(timezone.utc).replace(tzinfo=None)
    return ProductName(sat_id=sat_id, time=time, target=target, level=level.upper(), ac=ac, label=label)


def _parse_current(name: str) -> Optional[ProductName]:
    fields = name.split(".")
    if len(fields) < 5 or not _PLATFORM.fullmatch(fields[0]):
        return None
    platform, time, target, level, *rest, extension = fields
    if not _LEVEL.fullmatch(level) or not _TOKEN.fullmatch(target):
        return None
    try:
        time = datetime.strptime(time, _TIME_FORMAT)
    except ValueError:
        return None
    ac = label = None
    if level == "L2A":
        if not rest:
            return None
        ac, *rest = rest
    if len(rest) > 1 or not all(_TOKEN.fullmatch(f) for f in ([ac] if ac else []) + rest):
        return None
    if rest:
        label = rest[0]
    return ProductName(sat_id=sat_id_from_token(platform), time=time, target=target, level=level,
                       ac=ac, label=label, extension=extension)


def _parse_legacy(name: str) -> Optional[ProductName]:
    match = _LEGACY.fullmatch(name)
    if not match:
        return None
    return ProductName(sat_id=None,
                       time=datetime.strptime(match["time"], _CAPTURE_ID_TIME_FORMAT),
                       target=match["target"], level=match["level"].upper(), ac=match["ac"],
                       legacy_label=match["label"], extension=match["ext"], style="legacy")


def parse_product_name(path) -> ProductName:
    """Parse a current-style or legacy product file name."""
    name = Path(path).name
    parsed = _parse_current(name) or _parse_legacy(name)
    if parsed is None:
        raise ValueError(f"not a HYPSO product file name: {name}")
    return parsed
