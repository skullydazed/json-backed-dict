from __future__ import annotations

import contextlib
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
                raise TypeError(f'Dict key at {path} must be str, got {type(k).__name__!r}: {k!r}')
            _validate_json_value(v, f'{path}[{k!r}]')
        return
    raise TypeError(f'Value at {path} has unsupported type {type(value).__name__!r}: {value!r}')


# --- Exclusion helper ---


def _is_excluded(path: str, exclude_keys: set[str]) -> bool:
    """Return True if *path* should not trigger a write.

    A path is excluded if it exactly matches an entry in *exclude_keys* or if
    it falls under one (i.e. an excluded path is a dot-separated prefix of
    *path*).  Examples with ``exclude_keys={'session', 'config.timeout'}``::

        'session'          → True   (exact match)
        'session.user'     → True   (under excluded prefix 'session')
        'config.timeout'   → True   (exact match)
        'config.other'     → False
        'config'           → False  (a *parent* of an excluded path, not under one)
    """
    if not exclude_keys:
        return False
    if path in exclude_keys:
        return True
    for excl in exclude_keys:
        if path.startswith(excl + '.'):
            return True
    return False


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


def _wrap(value: Any, root: 'JsonBackedDict', path: str = '') -> Any:
    if isinstance(value, dict):
        return _ProxyDict(value, root, path)
    if isinstance(value, list):
        return _ProxyList(value, root, path)
    return value


class _ProxyDict:
    """Transparent proxy for a nested dict that propagates mutations to the root
    JsonBackedDict."""

    __slots__ = ('_data', '_root', '_path')
    __hash__ = None

    def __init__(self, data: dict, root: 'JsonBackedDict', path: str = '') -> None:
        self._data = data
        self._root = root
        self._path = path

    def _child_path(self, key: str) -> str:
        return f'{self._path}.{key}' if self._path else key

    def __getitem__(self, key: str) -> Any:
        with self._root._lock:
            return _wrap(self._data[key], self._root, self._child_path(key))

    def __setitem__(self, key: str, value: Any) -> None:
        with self._root._lock:
            if not isinstance(key, str):
                raise TypeError(f'Keys must be str, got {type(key).__name__!r}')
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data[key] = value
            if not _is_excluded(self._child_path(key), self._root._exclude_keys):
                self._root._save()

    def __delitem__(self, key: str) -> None:
        with self._root._lock:
            del self._data[key]
            if not _is_excluded(self._child_path(key), self._root._exclude_keys):
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
            return _wrap(self._data.get(key, default), self._root, self._child_path(key))

    def keys(self) -> Any:
        with self._root._lock:
            return list(self._data.keys())

    def values(self) -> Any:
        with self._root._lock:
            return [_wrap(v, self._root, self._child_path(k)) for k, v in self._data.items()]

    def items(self) -> Any:
        with self._root._lock:
            return [(k, _wrap(v, self._root, self._child_path(k))) for k, v in self._data.items()]

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
            if any(not _is_excluded(self._child_path(k), self._root._exclude_keys) for k in prepared):
                self._root._save()

    def pop(self, key: str, *args: Any) -> Any:
        with self._root._lock:
            if key not in self._data and args:
                return args[0]
            result = self._data.pop(key, *args)
            if not _is_excluded(self._child_path(key), self._root._exclude_keys):
                self._root._save()
            return result

    def setdefault(self, key: str, default: Any = None) -> Any:
        with self._root._lock:
            if key in self._data:
                return _wrap(self._data[key], self._root, self._child_path(key))
            default = _deep_unwrap(default)
            _validate_json_value(default)
            self._data[key] = default
            if not _is_excluded(self._child_path(key), self._root._exclude_keys):
                self._root._save()
            return _wrap(default, self._root, self._child_path(key))

    def popitem(self) -> tuple[str, Any]:
        with self._root._lock:
            k, v = self._data.popitem()
            if not _is_excluded(self._child_path(k), self._root._exclude_keys):
                self._root._save()
            return k, v

    def clear(self) -> None:
        with self._root._lock:
            self._data.clear()
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def save(self) -> None:
        """Explicitly flush to disk, bypassing batch mode and write_enabled."""
        with self._root._lock:
            self._root._save(force=True)


class _ProxyList:
    """Transparent proxy for a nested list that propagates mutations to the root
    JsonBackedDict."""

    __slots__ = ('_data', '_root', '_path')
    __hash__ = None

    def __init__(self, data: list, root: 'JsonBackedDict', path: str = '') -> None:
        self._data = data
        self._root = root
        self._path = path

    def __getitem__(self, idx: Any) -> Any:
        with self._root._lock:
            result = self._data[idx]
            if isinstance(idx, slice):
                return [_wrap(v, self._root, self._path) for v in result]
            return _wrap(result, self._root, self._path)

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
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def __delitem__(self, idx: Any) -> None:
        with self._root._lock:
            del self._data[idx]
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def __len__(self) -> int:
        with self._root._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        with self._root._lock:
            return iter([_wrap(v, self._root, self._path) for v in self._data])

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
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()
        return self

    def append(self, value: Any) -> None:
        with self._root._lock:
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data.append(value)
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def extend(self, values: Any) -> None:
        with self._root._lock:
            items = [_deep_unwrap(v) for v in values]
            for i, item in enumerate(items):
                _validate_json_value(item, f'root[{i}]')
            self._data.extend(items)
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def insert(self, idx: int, value: Any) -> None:
        with self._root._lock:
            value = _deep_unwrap(value)
            _validate_json_value(value)
            self._data.insert(idx, value)
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def remove(self, value: Any) -> None:
        with self._root._lock:
            self._data.remove(_deep_unwrap(value))
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def pop(self, idx: int = -1) -> Any:
        with self._root._lock:
            result = self._data.pop(idx)
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()
            return result

    def clear(self) -> None:
        with self._root._lock:
            self._data.clear()
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        with self._root._lock:
            self._data.sort(key=key, reverse=reverse)
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def reverse(self) -> None:
        with self._root._lock:
            self._data.reverse()
            if not _is_excluded(self._path, self._root._exclude_keys):
                self._root._save()

    def save(self) -> None:
        """Explicitly flush to disk, bypassing batch mode and write_enabled."""
        with self._root._lock:
            self._root._save(force=True)


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

    **Write control:** By default every mutation triggers an immediate disk
    write. Three opt-in mechanisms let callers take control:

    - ``write_enabled = False`` — suppresses all auto-saves; call
      :meth:`save` explicitly to flush.
    - :meth:`batch` context manager — defers writes across a block and
      produces at most one flush on exit (respects ``write_enabled``).
    - :meth:`exclude` / :meth:`include` — mutations to specific keys or
      dotted sub-key paths (e.g. ``'config.debug'``) do not trigger a write;
      the values are still serialized normally when any other write occurs.

    **Thread safety:** This class is thread-safe. Methods implemented by this
    class acquire an instance-level ``threading.RLock`` for their full duration,
    making individual operations atomic. Compound operations across multiple
    method calls are NOT atomic by default; callers needing compound atomicity
    may acquire ``instance._lock`` externally. Process-safety is not provided:
    concurrent writes from separate processes can still race, and external
    modifications to the backing file are not detected by a running instance.
    To see external changes, construct a new ``JsonBackedDict`` from the same
    path.
    """

    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path)
        self._deferred_depth: int = 0
        self._dirty: bool = False
        self._exclude_keys: set[str] = set()
        self.write_enabled: bool = True
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

    def _save(self, force: bool = False) -> None:
        if not force and (self._deferred_depth > 0 or not self.write_enabled):
            if self._deferred_depth > 0:
                self._dirty = True
            return
        self._dirty = False
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
            return _wrap(super().__getitem__(key), self, key)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            return _wrap(super().get(key, default), self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if not isinstance(key, str):
                raise TypeError(f'Keys must be str, got {type(key).__name__!r}')
            value = _deep_unwrap(value)
            _validate_json_value(value)
            super().__setitem__(key, value)
            if not _is_excluded(key, self._exclude_keys):
                self._save()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            super().__delitem__(key)
            if not _is_excluded(key, self._exclude_keys):
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
            if any(not _is_excluded(k, self._exclude_keys) for k in prepared):
                self._save()

    def __ior__(self, other: Any) -> 'JsonBackedDict':
        self.update(other)
        return self

    def __or__(self, other: Any) -> dict:  # type: ignore[override]
        with self._lock:
            if not isinstance(other, dict):
                return NotImplemented  # type: ignore[return-value]
            merged = {k: _deep_unwrap(dict.__getitem__(self, k)) for k in dict.__iter__(self)}
            merged.update(other)
            return merged

    def __ror__(self, other: Any) -> dict:
        with self._lock:
            if not isinstance(other, dict):
                return NotImplemented  # type: ignore[return-value]
            merged = dict(other)
            merged.update({k: _deep_unwrap(dict.__getitem__(self, k)) for k in dict.__iter__(self)})
            return merged

    def pop(self, key: str, *args: Any) -> Any:  # type: ignore[override]
        with self._lock:
            if key not in self and args:
                return args[0]
            result = super().pop(key, *args)
            if not _is_excluded(key, self._exclude_keys):
                self._save()
            return result

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._save()

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            if key in self:
                return _wrap(super().__getitem__(key), self, key)
            default = _deep_unwrap(default)
            _validate_json_value(default)
            super().__setitem__(key, default)
            if not _is_excluded(key, self._exclude_keys):
                self._save()
            return _wrap(default, self, key)

    def popitem(self) -> tuple[str, Any]:
        with self._lock:
            result = super().popitem()
            self._save()
            return result

    def keys(self) -> Any:  # type: ignore[override]
        with self._lock:
            return list(dict.__iter__(self))

    def __reversed__(self):
        with self._lock:
            return reversed(list(dict.__iter__(self)))

    def values(self) -> Any:  # type: ignore[override]
        with self._lock:
            return [_wrap(dict.__getitem__(self, k), self, k) for k in dict.__iter__(self)]

    def items(self) -> Any:  # type: ignore[override]
        with self._lock:
            return [(k, _wrap(dict.__getitem__(self, k), self, k)) for k in dict.__iter__(self)]

    def copy(self) -> dict:  # type: ignore[override]
        # Returns a plain dict with raw (unwrapped) values; proxy objects are
        # an implementation detail of JsonBackedDict and are not exposed here.
        # Nested mutable containers are deep-unwrapped so that callers cannot
        # mutate the internal backing structures via the returned dict.
        with self._lock:
            return {k: _deep_unwrap(dict.__getitem__(self, k)) for k in dict.__iter__(self)}

    def __repr__(self) -> str:
        with self._lock:
            raw = {k: dict.__getitem__(self, k) for k in dict.__iter__(self)}
            return f'{type(self).__name__}({self._path!r}, {raw!r})'

    @contextlib.contextmanager
    def batch(self):
        """Context manager that defers disk writes until the outermost block exits.

        Multiple mutations inside the block produce at most one file write on
        exit. Nested ``batch()`` calls are safe — only the outermost block
        triggers the flush::

            with d.batch():
                d['a'] = 1
                d['nested']['key'] = 2  # proxy mutations also deferred
            # flushed here only if mutations occurred, and only if write_enabled

        This context manager does **not** hold ``self._lock`` for the full
        duration of the block. Other threads may still read or mutate the
        object concurrently; each individual operation remains atomic as usual.
        Batch semantics only defer flushing to disk until the outermost batch
        exits.

        If an exception propagates out of the block, exit handling still
        attempts a flush so successful mutations are not silently lost, subject
        to ``write_enabled``.
        """
        with self._lock:
            self._deferred_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._deferred_depth -= 1
                if self._deferred_depth == 0 and self._dirty:
                    self._save()

    def exclude(self, key: str) -> None:
        """Suppress writes triggered by mutations to *key*.

        *key* may be a dotted path, e.g. ``'config.timeout'``. Mutations to
        that path (or any path beneath it) will not trigger a disk write.
        The value is still included in the file whenever any other write
        occurs. Call :meth:`include` to restore normal write-on-mutate
        behaviour.
        """
        if not isinstance(key, str):
            raise TypeError(f'key must be str, got {type(key).__name__!r}')
        with self._lock:
            self._exclude_keys.add(key)

    def include(self, key: str) -> None:
        """Re-enable write-on-mutate for a previously excluded key or path."""
        if not isinstance(key, str):
            raise TypeError(f'key must be str, got {type(key).__name__!r}')
        with self._lock:
            self._exclude_keys.discard(key)

    def save(self) -> None:
        """Explicitly flush all keys to disk.

        Bypasses both batch deferral and ``write_enabled`` — use this when
        you need a guaranteed write regardless of the current write mode.
        All keys, including those passed to :meth:`exclude`, are written.
        """
        with self._lock:
            self._save(force=True)

    # Prevent pickle/copy from bypassing __init__ and missing _path.
    # Unpickling calls __init__(path), which reloads from the file. We do not
    # pass the in-memory dict as `initial` because the file always reflects the
    # current state and `initial` would be silently ignored anyway (the file
    # exists at that point).
    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self._path,))
