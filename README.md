# json-backed-dict

A `dict` subclass that automatically persists every mutation to a JSON file.

```python
from json_backed_dict import JsonBackedDict as JBD

d = JBD('config.json')
d['host'] = 'localhost'   # written to disk immediately
d['port'] = 5432          # written to disk immediately
```

## Features

- **Automatic persistence** — every `__setitem__`, `__delitem__`, `update`, `pop`, etc. atomically writes the file
- **Nested mutation tracking** — mutating a nested dict or list is also persisted
- **Temporal type round-tripping** — `datetime`, `date`, `time`, and `timedelta` values survive save/load cycles as their Python types
- **Atomic writes** — uses a temp file + `os.replace` so a crash mid-write never corrupts the file
- **Fast** — backed by [orjson](https://github.com/ijl/orjson) for serialization

## Installation

```
pip install json-backed-dict
```

Requires Python 3.9+.

## Usage

### Basic operations

```python
from json_backed_dict import JsonBackedDict as JBD

d = JBD('data.json')

d['name'] = 'Alice'
d['scores'] = [10, 20, 30]
d.update({'active': True, 'level': 5})

print(d['name'])      # 'Alice'
print(d.get('age'))   # None
del d['level']
```

All standard `dict` methods work: `keys()`, `values()`, `items()`, `pop()`, `setdefault()`, `popitem()`, `clear()`, `update()`.

### Loading an existing file

If the file already exists, it is loaded on construction. The `initial` argument is ignored when a file is present.

```python
d = JBD('data.json')          # loads existing file
d = JBD('data.json', initial={'x': 1})  # initial ignored, file loaded
```

### Seeding a new file

```python
d = JBD('settings.json', initial={'debug': False, 'timeout': 30})
```

`initial` is only used when creating a new file.

### Nested mutations

Nested dicts and lists returned by `__getitem__` and `get()` are proxy objects. Mutating them persists the change to the root file automatically.

```python
d = JBD('data.json', initial={'config': {'timeout': 10}, 'tags': ['a', 'b']})

d['config']['timeout'] = 30   # persisted
d['tags'].append('c')         # persisted
d['config'].update({'retries': 3})  # persisted
```

### Temporal types

`datetime`, `date`, `time`, and `timedelta` values are serialized to strings and deserialized back to their Python types on load. No manual conversion needed.

```python
from datetime import datetime, date, time, timedelta

d = JBD('data.json')
d['created_at'] = datetime(2024, 6, 15, 10, 30, 45)
d['due_date']   = date(2024, 7, 1)
d['start_time'] = time(9, 0)
d['ttl']        = timedelta(hours=24)

d2 = JBD('data.json')
isinstance(d2['created_at'], datetime)   # True
isinstance(d2['ttl'], timedelta)         # True
```

**Note:** This means strings that look like dates, times, or timedeltas cannot be stored as plain strings — they will be coerced to the corresponding Python type on the next load. For example, storing `"2024-01-15"` will be read back as `date(2024, 1, 15)`.

### Supported value types

| Type | Example |
|------|---------|
| `str` | `'hello'` |
| `int` | `42` |
| `float` | `3.14` |
| `bool` | `True` |
| `None` | `None` |
| `list` | `[1, 2, 3]` |
| `dict` | `{'key': 'value'}` |
| `datetime` | `datetime(2024, 1, 1, 12, 0)` |
| `date` | `date(2024, 1, 1)` |
| `time` | `time(12, 0, 0)` |
| `timedelta` | `timedelta(days=1, hours=2)` |

All dict keys must be `str`. Attempting to store any other type raises `TypeError` before any mutation occurs.

## Limitations

**Thread-safe, but not process-safe.** Methods implemented by `JsonBackedDict` acquire an instance-level lock for their full duration, making individual operations on a single instance atomic. Concurrent writes from multiple processes can still interleave: each write creates a temp file and calls `os.replace`, so the last writer wins and earlier changes are silently lost.

**Not an IPC mechanism.** If another process modifies the backing file, the current instance will never see those changes. To pick up external changes, construct a new `JBD` from the same path.

## License

MIT
