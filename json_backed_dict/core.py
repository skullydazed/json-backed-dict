from __future__ import annotations

import os
import re
import tempfile
import threading
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


# --- Proxy unwrap helper ---


def _deep_unwrap(value: Any) -> Any:
    """Recursively strip _ProxyDict/_ProxyList wrappers, returning plain dicts/lists."""
    if isinstance(value, _ProxyDict):
        return {k: _deep_unwrap(v) for k, v in value._data.items()}
    if isinstance(value, _ProxyList):
        return [_deep_unwrap(v) for v in value._data]
    if isinstance(value, dict):
        return {k: _deep_unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_unwrap(v) for v in value]
    return value


# --- Atomic write ---


def _atomic_write(dest: Path, data: bytes) -> None:
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix='.tmp')
        # os.fdopen takes ownership of fd; close it manually only if fdopen fails.
        try:
            f = os.fdopen(fd, 'wb')
        except Exception:
            os.close(fd)
            raise
        with f:
            f.write(data)
        os.replace(tmp_path, dest)
        tmp_path = None  # replaced successfully, nothing to clean up
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# --- Proxy classes ---


def _wrap(value: Any, root: 'JsonBackedDict') -> Any:
    if isinstance(value, dict):
        return _ProxyDict(value, root)
    if isinstance(value, list):
        return _ProxyList(value, root)
    return value


class _ProxyDict:
    """Transparent proxy for a nested dict that propagates mutations to the root
    JsonBackedDict."""

    __slots__ = ('_data', '_root')

    def __init__(self, data: dict, root: 'JsonBackedDict') -> None:
        self._data = data
        self._root = root

    def __getitem__(self, key: str) -> Any:
        with self._root._lock:
            return _wrap(self._data[key], self._root)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._root._lock:
            if not isinstance(key, str):
                raise TypeError(f'Keys must be str, got {type(key).__name__!r}')
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data[key] = value
            self._root._save()

    def __delitem__(self, key: str) -> None:
        with self._root._lock:
            del self._data[key]
            self._root._save()

    def __contains__(self, key: object) -> bool:
        with self._root._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._root._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[str]:
        with self._root._lock:
            return iter(list(self._data))

    def __eq__(self, other: object) -> bool:
        with self._root._lock:
            if isinstance(other, _ProxyDict):
                return self._data == other._data
            return self._data == other

    def __repr__(self) -> str:
        with self._root._lock:
            return repr(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        with self._root._lock:
            return _wrap(self._data.get(key, default), self._root)

    def keys(self) -> Any:
        with self._root._lock:
            return list(self._data.keys())

    def values(self) -> Any:
        with self._root._lock:
            return [_wrap(v, self._root) for v in self._data.values()]

    def items(self) -> Any:
        with self._root._lock:
            return [(k, _wrap(v, self._root)) for k, v in self._data.items()]

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        with self._root._lock:
            merged = dict(other)
            merged.update(kwargs)
            if not merged:
                return
            prepared: dict[str, Any] = {}
            for k, v in merged.items():
                if not isinstance(k, str):
                    raise TypeError(f'Keys must be str, got {type(k).__name__!r}: {k!r}')
                u = _deep_unwrap(v)
                _validate_json_value(u)
                prepared[k] = u
            for k, v in prepared.items():
                self._data[k] = v
            self._root._save()

    def pop(self, key: str, *args: Any) -> Any:
        with self._root._lock:
            if key not in self._data and args:
                return args[0]
            result = self._data.pop(key, *args)
            self._root._save()
            return result

    def setdefault(self, key: str, default: Any = None) -> Any:
        with self._root._lock:
            if key in self._data:
                return _wrap(self._data[key], self._root)
            default = _deep_unwrap(default)
            _validate_json_value(default)
            self._data[key] = default
            self._root._save()
            return _wrap(default, self._root)

    def popitem(self) -> tuple[str, Any]:
        with self._root._lock:
            result = self._data.popitem()
            self._root._save()
            return result

    def clear(self) -> None:
        with self._root._lock:
            self._data.clear()
            self._root._save()


class _ProxyList:
    """Transparent proxy for a nested list that propagates mutations to the root
    JsonBackedDict."""

    __slots__ = ('_data', '_root')

    def __init__(self, data: list, root: 'JsonBackedDict') -> None:
        self._data = data
        self._root = root

    def __getitem__(self, idx: Any) -> Any:
        with self._root._lock:
            result = self._data[idx]
            if isinstance(idx, slice):
                return [_wrap(v, self._root) for v in result]
            return _wrap(result, self._root)

    def __setitem__(self, idx: Any, value: Any) -> None:
        with self._root._lock:
            if isinstance(idx, slice):
                items = [_deep_unwrap(v) for v in value]
                for i, item in enumerate(items):
                    _validate_json_value(item, f'root[{i}]')
                self._data[idx] = items
            else:
                value = _deep_unwrap(value)
                _validate_json_value(value)
                self._data[idx] = value
            self._root._save()

    def __delitem__(self, idx: Any) -> None:
        with self._root._lock:
            del self._data[idx]
            self._root._save()

    def __len__(self) -> int:
        with self._root._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        with self._root._lock:
            return iter([_wrap(v, self._root) for v in self._data])

    def __contains__(self, item: object) -> bool:
        with self._root._lock:
            return _deep_unwrap(item) in self._data

    def __eq__(self, other: object) -> bool:
        with self._root._lock:
            if isinstance(other, _ProxyList):
                return self._data == other._data
            return self._data == other

    def __repr__(self) -> str:
        with self._root._lock:
            return repr(self._data)

    def __iadd__(self, other: Any) -> '_ProxyList':
        self.extend(other)
        return self

    def __imul__(self, n: int) -> '_ProxyList':
        with self._root._lock:
            self._data *= n
            self._root._save()
        return self

    def append(self, value: Any) -> None:
        with self._root._lock:
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data.append(value)
            self._root._save()

    def extend(self, values: Any) -> None:
        with self._root._lock:
            items = [_deep_unwrap(v) for v in values]
            for i, item in enumerate(items):
                _validate_json_value(item, f'root[{i}]')
            self._data.extend(items)
            self._root._save()

    def insert(self, idx: int, value: Any) -> None:
        with self._root._lock:
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data.insert(idx, value)
            self._root._save()

    def remove(self, value: Any) -> None:
        with self._root._lock:
            self._data.remove(_deep_unwrap(value))
            self._root._save()

    def pop(self, idx: int = -1) -> Any:
        with self._root._lock:
            result = self._data.pop(idx)
            self._root._save()
            return result

    def clear(self) -> None:
        with self._root._lock:
            self._data.clear()
            self._root._save()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        with self._root._lock:
            self._data.sort(key=key, reverse=reverse)
            self._root._save()

    def reverse(self) -> None:
        with self._root._lock:
            self._data.reverse()
            self._root._save()


# --- Main class ---


class JsonBackedDict(dict):  # type: ignore[type-arg]
    """A dict subclass that persists all mutations to a JSON file atomically.

    String values that match ISO 8601 date/time patterns or the timedelta format
    are automatically converted to their Python types on load. This means you
    cannot store plain strings that look like dates, times, or timedeltas — they
    will round-trip as the corresponding Python objects:

        - ``"2024-01-15"``             → ``datetime.date(2024, 1, 15)``
        - ``"2024-01-15T10:30:00"``    → ``datetime.datetime(2024, 1, 15, 10, 30)``
        - ``"10:30:00"``               → ``datetime.time(10, 30)``
        - ``"1d2h30m"``                → ``datetime.timedelta(days=1, hours=2,
          minutes=30)``

    Nested dict and list mutations are automatically persisted via proxy objects
    returned by ``__getitem__`` and ``get()``. Mutating a nested value triggers
    a full file save:

        d['config']['timeout'] = 30   # persisted
        d['items'].append('new')      # persisted

    **Thread safety:** This class is thread-safe. Each public method acquires an
    instance-level ``threading.RLock`` for its full duration, making individual
    operations atomic. Compound operations across multiple method calls are NOT
    atomic by default; callers needing compound atomicity may acquire
    ``instance._lock`` externally. Process-safety is not provided: concurrent
    writes from separate processes can still race, and external modifications to
    the backing file are not detected by a running instance. To see external
    changes, construct a new ``JsonBackedDict`` from the same path.
    """

    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path)
        if self._path.exists():
            try:
                raw = orjson.loads(self._path.read_bytes())
            except Exception as e:
                raise ValueError(f'Failed to load {self._path}: {e}') from e
            super().__init__(_decode_value(raw))
        else:
            if initial is not None:
                _validate_json_value(initial)
            super().__init__(initial or {})
            self._save()

    def _save(self) -> None:
        # Use dict.__getitem__/__iter__ directly to bypass the proxy-wrapping
        # __getitem__ override, ensuring raw values are serialized.
        raw = {k: dict.__getitem__(self, k) for k in dict.__iter__(self)}
        data = orjson.dumps(
            raw,
            default=_orjson_default,
            option=orjson.OPT_INDENT_2,
        )
        _atomic_write(self._path, data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return super().__contains__(key)

    def __len__(self) -> int:
        with self._lock:
            return super().__len__()

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(super().__iter__()))

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return _wrap(super().__getitem__(key), self)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            return _wrap(super().get(key, default), self)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if not isinstance(key, str):
                raise TypeError(f'Keys must be str, got {type(key).__name__!r}')
            value = _deep_unwrap(value)
            _validate_json_value(value)
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            super().__delitem__(key)
            self._save()

    def update(self, other: Any = (), /, **kwargs: Any) -> None:  # type: ignore[override]
        with self._lock:
            merged = dict(other)
            merged.update(kwargs)
            if not merged:
                return
            prepared: dict[str, Any] = {}
            for k, v in merged.items():
                if not isinstance(k, str):
                    raise TypeError(f'Keys must be str, got {type(k).__name__!r}: {k!r}')
                u = _deep_unwrap(v)
                _validate_json_value(u)
                prepared[k] = u
            for k, v in prepared.items():
                super().__setitem__(k, v)
            self._save()

    def __ior__(self, other: Any) -> 'JsonBackedDict':
        self.update(other)
        return self

    def __or__(self, other: Any) -> dict:  # type: ignore[override]
        with self._lock:
            if not isinstance(other, dict):
                return NotImplemented  # type: ignore[return-value]
            merged = dict(self)
            merged.update(other)
            return merged

    def __ror__(self, other: Any) -> dict:
        with self._lock:
            if not isinstance(other, dict):
                return NotImplemented  # type: ignore[return-value]
            merged = dict(other)
            merged.update(self)
            return merged

    def pop(self, key: str, *args: Any) -> Any:  # type: ignore[override]
        with self._lock:
            if key not in self and args:
                return args[0]
            result = super().pop(key, *args)
            self._save()
            return result

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._save()

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            if key in self:
                return _wrap(super().__getitem__(key), self)
            default = _deep_unwrap(default)
            _validate_json_value(default)
            super().__setitem__(key, default)
            self._save()
            return _wrap(default, self)

    def popitem(self) -> tuple[str, Any]:
        with self._lock:
            result = super().popitem()
            self._save()
            return result

    def values(self) -> Any:  # type: ignore[override]
        with self._lock:
            return [_wrap(v, self) for v in super().values()]

    def items(self) -> Any:  # type: ignore[override]
        with self._lock:
            return [(k, _wrap(v, self)) for k, v in super().items()]

    def __repr__(self) -> str:
        with self._lock:
            return f'{type(self).__name__}({self._path!r}, {dict(self)!r})'

    # Prevent pickle/copy from bypassing __init__ and missing _path.
    # Unpickling calls __init__(path), which reloads from the file. We do not
    # pass the in-memory dict as `initial` because the file always reflects the
    # current state and `initial` would be silently ignored anyway (the file
    # exists at that point).
    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self._path,))
