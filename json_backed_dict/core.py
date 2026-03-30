from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterator

import orjson

# --- Compiled patterns for temporal string detection ---

_DATETIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_TIME_RE = re.compile(r'^\d{2}:\d{2}(:\d{2}(\.\d+)?)?$')
_TIMEDELTA_RE = re.compile(
    r'^(?P<neg>-)?(?:(?P<d>\d+)d)?(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s)?(?:(?P<us>\d+)us)?$'
)

# --- Timedelta helpers ---


def _timedelta_to_str(td: timedelta) -> str:
    negative = td.days < 0
    if negative:
        td = -td

    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    microseconds = td.microseconds

    parts = []
    if days:
        parts.append(f'{days}d')
    if hours:
        parts.append(f'{hours}h')
    if minutes:
        parts.append(f'{minutes}m')
    if secs:
        parts.append(f'{secs}s')
    if microseconds:
        parts.append(f'{microseconds}us')

    result = ''.join(parts) or '0s'
    return f'-{result}' if negative else result


def _str_to_timedelta(s: str) -> timedelta:
    m = _TIMEDELTA_RE.match(s)
    if not m or not any(m.group(k) for k in ('d', 'h', 'm', 's', 'us')):
        raise ValueError(f'Invalid timedelta string: {s!r}')

    td = timedelta(
        days=int(m.group('d') or 0),
        hours=int(m.group('h') or 0),
        minutes=int(m.group('m') or 0),
        seconds=int(m.group('s') or 0),
        microseconds=int(m.group('us') or 0),
    )
    return -td if m.group('neg') else td


# --- orjson custom serializer ---


def _orjson_default(obj: Any) -> Any:
    if isinstance(obj, timedelta):
        return _timedelta_to_str(obj)
    raise TypeError(f'Object of type {type(obj).__name__!r} is not JSON serializable')


# --- Decode helpers ---


def _try_parse_temporal(s: str) -> Any:
    if _DATETIME_RE.match(s):
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
    if _DATE_RE.match(s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
    if _TIME_RE.match(s):
        try:
            return time.fromisoformat(s)
        except ValueError:
            pass
    m = _TIMEDELTA_RE.match(s)
    if m and any(m.group(k) for k in ('d', 'h', 'm', 's', 'us')):
        return _str_to_timedelta(s)
    return s


def _decode_value(v: Any) -> Any:
    if isinstance(v, str):
        return _try_parse_temporal(v)
    if isinstance(v, dict):
        return {k: _decode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_decode_value(item) for item in v]
    return v


# --- Validation ---

_VALID_LEAF_TYPES = (str, bool, int, float, type(None), datetime, date, time, timedelta)


def _validate_json_value(value: Any, path: str = 'root') -> None:
    # Check bool before int (bool is subclass of int)
    # Check datetime before date (datetime is subclass of date)
    _leaf_types = (bool, str, int, float, type(None), datetime, date, time, timedelta)
    if isinstance(value, _leaf_types):
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate_json_value(item, f'{path}[{i}]')
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    f'Dict key at {path} must be str, got {type(k).__name__!r}: {k!r}'
                )
            _validate_json_value(v, f'{path}[{k!r}]')
        return
    raise TypeError(
        f'Value at {path} has unsupported type {type(value).__name__!r}: {value!r}'
    )


# --- Atomic write ---


def _atomic_write(dest: Path, data: bytes) -> None:
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
        except Exception:
            os.close(fd)
            raise
        os.replace(tmp_path, dest)
        tmp_path = None  # replaced successfully, nothing to clean up
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# --- Main class ---


class JsonBackedDict(dict):  # type: ignore[type-arg]
    """A dict subclass that persists all mutations to a JSON file atomically."""

    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None) -> None:
        self._path = Path(path)
        if self._path.exists():
            raw = orjson.loads(self._path.read_bytes())
            super().__init__(_decode_value(raw))
        else:
            if initial is not None:
                _validate_json_value(initial)
            super().__init__(initial or {})
            self._save()

    def _save(self) -> None:
        data = orjson.dumps(
            dict(self),
            default=_orjson_default,
            option=orjson.OPT_INDENT_2,
        )
        _atomic_write(self._path, data)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f'Keys must be str, got {type(key).__name__!r}')
        _validate_json_value(value)
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._save()

    def update(self, other: Any = (), /, **kwargs: Any) -> None:  # type: ignore[override]
        merged = dict(other)
        merged.update(kwargs)
        for k, v in merged.items():
            if not isinstance(k, str):
                raise TypeError(f'Keys must be str, got {type(k).__name__!r}: {k!r}')
            _validate_json_value(v)
        super().update(merged)
        self._save()

    def pop(self, key: str, *args: Any) -> Any:  # type: ignore[override]
        result = super().pop(key, *args)
        self._save()
        return result

    def clear(self) -> None:
        super().clear()
        self._save()

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            return self[key]
        _validate_json_value(default)
        super().__setitem__(key, default)
        self._save()
        return default

    def popitem(self) -> tuple[str, Any]:
        result = super().popitem()
        self._save()
        return result

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._path!r}, {dict(self)!r})'

    # Prevent pickle/copy from bypassing __init__ and missing _path
    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self._path, dict(self)))

    # Make iteration, keys, values, items work naturally via dict
    def keys(self) -> Any:
        return super().keys()

    def values(self) -> Any:
        return super().values()

    def items(self) -> Any:
        return super().items()

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def __len__(self) -> int:
        return super().__len__()

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key)
