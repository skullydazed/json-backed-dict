from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import orjson
import pytest

from json_backed_dict import JsonBackedDict
from json_backed_dict.core import _str_to_timedelta, _timedelta_to_str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_raw(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_creates_file_when_not_exists(self, tmp_path):
        p = tmp_path / 'data.json'
        JsonBackedDict(p)
        assert p.exists()
        assert load_raw(p) == {}

    def test_empty_dict_in_memory(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        assert dict(d) == {}

    def test_initial_dict_used_when_no_file(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1, 'b': 'hello'})
        assert d['a'] == 1
        assert d['b'] == 'hello'
        assert load_raw(p) == {'a': 1, 'b': 'hello'}

    def test_initial_ignored_when_file_exists(self, tmp_path):
        p = tmp_path / 'data.json'
        p.write_bytes(orjson.dumps({'x': 10}))
        d = JsonBackedDict(p, initial={'y': 20})
        assert dict(d) == {'x': 10}

    def test_loads_existing_file(self, tmp_path):
        p = tmp_path / 'data.json'
        p.write_bytes(orjson.dumps({'foo': 'bar', 'n': 42}))
        d = JsonBackedDict(p)
        assert d['foo'] == 'bar'
        assert d['n'] == 42

    def test_initial_validates_types(self, tmp_path):
        with pytest.raises(TypeError):
            JsonBackedDict(tmp_path / 'data.json', initial={'bad': object()})

    def test_accepts_path_as_string(self, tmp_path):
        p = str(tmp_path / 'data.json')
        JsonBackedDict(p)
        assert Path(p).exists()


# ---------------------------------------------------------------------------
# 2. Basic CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_setitem_stores_in_memory(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        d['key'] = 'value'
        assert d['key'] == 'value'

    def test_setitem_persists_to_file(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['key'] = 'value'
        assert load_raw(p)['key'] == 'value'

    def test_delitem_removes_key(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        del d['a']
        assert 'a' not in d

    def test_delitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        del d['a']
        assert 'a' not in load_raw(p)

    def test_delitem_missing_key_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(KeyError):
            del d['missing']

    def test_getitem_missing_key_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(KeyError):
            _ = d['missing']

    def test_non_string_key_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d[42] = 'value'  # type: ignore


# ---------------------------------------------------------------------------
# 3. Mutation methods
# ---------------------------------------------------------------------------


class TestMutationMethods:
    def test_update_with_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d.update({'a': 1, 'b': 2})
        assert d['a'] == 1 and d['b'] == 2
        raw = load_raw(p)
        assert raw['a'] == 1 and raw['b'] == 2

    def test_update_with_kwargs(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        d.update(x=10, y=20)
        assert d['x'] == 10 and d['y'] == 20

    def test_update_with_iterable_of_pairs(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        d.update([('a', 1), ('b', 2)])
        assert d['a'] == 1 and d['b'] == 2

    def test_pop_existing_key(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        result = d.pop('a')
        assert result == 1
        assert 'a' not in d
        assert 'a' not in load_raw(p)

    def test_pop_missing_with_default(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        result = d.pop('missing', 'fallback')
        assert result == 'fallback'

    def test_pop_missing_no_default_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(KeyError):
            d.pop('missing')

    def test_clear(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1, 'b': 2})
        d.clear()
        assert dict(d) == {}
        assert load_raw(p) == {}

    def test_setdefault_key_absent(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        result = d.setdefault('k', 'default')
        assert result == 'default'
        assert d['k'] == 'default'
        assert load_raw(p)['k'] == 'default'

    def test_setdefault_key_present(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'k': 'existing'})
        mtime_before = p.stat().st_mtime_ns
        result = d.setdefault('k', 'other')
        assert result == 'existing'
        assert p.stat().st_mtime_ns == mtime_before

    def test_popitem_returns_pair(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        key, val = d.popitem()
        assert key == 'a' and val == 1
        assert 'a' not in d
        assert load_raw(p) == {}

    def test_popitem_empty_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(KeyError):
            d.popitem()


# ---------------------------------------------------------------------------
# 4. Read operations
# ---------------------------------------------------------------------------


class TestReadOps:
    def test_contains(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        assert 'a' in d
        assert 'b' not in d

    def test_len(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1, 'b': 2})
        assert len(d) == 2

    def test_iter(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1, 'b': 2})
        assert set(d) == {'a', 'b'}

    def test_keys(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1, 'b': 2})
        assert set(d.keys()) == {'a', 'b'}

    def test_values(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1, 'b': 2})
        assert set(d.values()) == {1, 2}

    def test_items(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        assert list(d.items()) == [('a', 1)]

    def test_get_existing(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        assert d.get('a') == 1

    def test_get_missing_with_default(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        assert d.get('missing', 99) == 99

    def test_get_missing_no_default(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        assert d.get('missing') is None


# ---------------------------------------------------------------------------
# 5. Type validation
# ---------------------------------------------------------------------------


class TestTypeValidation:
    @pytest.mark.parametrize(
        'value',
        [
            'hello',
            42,
            3.14,
            True,
            False,
            None,
            [1, 2, 3],
            {'nested': 'dict'},
            datetime(2024, 1, 1, 12, 0),
            date(2024, 1, 1),
            time(12, 0),
            timedelta(days=1, hours=2),
        ],
    )
    def test_valid_types_accepted(self, tmp_path, value):
        d = JsonBackedDict(tmp_path / 'data.json')
        d['k'] = value  # should not raise

    @pytest.mark.parametrize(
        'value',
        [
            object(),
            b'bytes',
            {1, 2, 3},
            (1, 2),
        ],
    )
    def test_invalid_types_raise(self, tmp_path, value):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d['k'] = value

    def test_nested_dict_invalid_value_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d['k'] = {'nested': object()}

    def test_nested_list_invalid_value_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d['k'] = [1, 2, object()]

    def test_deeply_nested_invalid_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d['k'] = {'a': {'b': {'c': object()}}}

    def test_non_string_nested_dict_key_raises(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError):
            d['k'] = {1: 'val'}

    def test_update_validates_before_mutating(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        with pytest.raises(TypeError):
            d.update({'b': 2, 'c': object()})
        assert dict(d) == {'a': 1}
        assert load_raw(p) == {'a': 1}

    def test_error_message_includes_path(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json')
        with pytest.raises(TypeError, match='root'):
            d['k'] = object()


# ---------------------------------------------------------------------------
# 6. Datetime round-trip
# ---------------------------------------------------------------------------


class TestDatetimeRoundTrip:
    def _reload(self, path: Path) -> JsonBackedDict:
        return JsonBackedDict(path)

    def test_naive_datetime(self, tmp_path):
        p = tmp_path / 'data.json'
        dt = datetime(2024, 6, 15, 10, 30, 45)
        JsonBackedDict(p, initial={'dt': dt})
        d2 = self._reload(p)
        assert d2['dt'] == dt
        assert isinstance(d2['dt'], datetime)

    def test_tz_aware_datetime(self, tmp_path):
        p = tmp_path / 'data.json'
        dt = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        JsonBackedDict(p, initial={'dt': dt})
        d2 = self._reload(p)
        assert d2['dt'] == dt
        assert isinstance(d2['dt'], datetime)

    def test_date(self, tmp_path):
        p = tmp_path / 'data.json'
        v = date(2024, 6, 15)
        JsonBackedDict(p, initial={'d': v})
        d2 = self._reload(p)
        assert d2['d'] == v
        assert type(d2['d']) is date  # not datetime

    def test_time(self, tmp_path):
        p = tmp_path / 'data.json'
        v = time(10, 30, 45)
        JsonBackedDict(p, initial={'t': v})
        d2 = self._reload(p)
        assert d2['t'] == v
        assert isinstance(d2['t'], time)

    def test_datetime_not_confused_with_date(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['dt'] = datetime(2024, 1, 1, 0, 0, 0)
        d['da'] = date(2024, 1, 1)
        d2 = self._reload(p)
        assert isinstance(d2['dt'], datetime)
        assert type(d2['da']) is date

    def test_datetime_in_nested_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        dt = datetime(2024, 1, 1, 12, 0)
        JsonBackedDict(p, initial={'nested': {'ts': dt}})
        d2 = self._reload(p)
        assert d2['nested']['ts'] == dt

    def test_datetime_in_list(self, tmp_path):
        p = tmp_path / 'data.json'
        dates = [date(2024, 1, i) for i in range(1, 4)]
        JsonBackedDict(p, initial={'dates': dates})
        d2 = self._reload(p)
        assert d2['dates'] == dates


# ---------------------------------------------------------------------------
# 7. Timedelta round-trip
# ---------------------------------------------------------------------------


class TestTimedeltaRoundTrip:
    def _reload(self, path: Path) -> JsonBackedDict:
        return JsonBackedDict(path)

    @pytest.mark.parametrize(
        'td,expected_str',
        [
            (timedelta(0), '0s'),
            (timedelta(days=1), '1d'),
            (timedelta(hours=2), '2h'),
            (timedelta(minutes=30), '30m'),
            (timedelta(seconds=45), '45s'),
            (timedelta(days=1, hours=2, minutes=3, seconds=4), '1d2h3m4s'),
            (timedelta(microseconds=500), '500us'),
            (timedelta(hours=1, microseconds=100), '1h100us'),
        ],
    )
    def test_to_str(self, td, expected_str):
        assert _timedelta_to_str(td) == expected_str

    @pytest.mark.parametrize(
        's,expected_td',
        [
            ('0s', timedelta(0)),
            ('1d', timedelta(days=1)),
            ('2h', timedelta(hours=2)),
            ('30m', timedelta(minutes=30)),
            ('45s', timedelta(seconds=45)),
            ('1d2h3m4s', timedelta(days=1, hours=2, minutes=3, seconds=4)),
            ('500us', timedelta(microseconds=500)),
            ('1h100us', timedelta(hours=1, microseconds=100)),
        ],
    )
    def test_from_str(self, s, expected_td):
        assert _str_to_timedelta(s) == expected_td

    def test_negative_timedelta_to_str(self):
        assert _timedelta_to_str(timedelta(hours=-1)) == '-1h'
        assert _timedelta_to_str(timedelta(days=-2, hours=-3)) == '-2d3h'
        assert _timedelta_to_str(timedelta(microseconds=-500)) == '-500us'

    def test_negative_timedelta_from_str(self):
        assert _str_to_timedelta('-1h') == timedelta(hours=-1)
        assert _str_to_timedelta('-2d3h') == timedelta(days=-2, hours=-3)
        assert _str_to_timedelta('-500us') == timedelta(microseconds=-500)

    def test_round_trip_simple(self, tmp_path):
        p = tmp_path / 'data.json'
        td = timedelta(days=1, hours=2, minutes=3, seconds=4)
        JsonBackedDict(p, initial={'td': td})
        d2 = self._reload(p)
        assert d2['td'] == td
        assert isinstance(d2['td'], timedelta)

    def test_round_trip_zero(self, tmp_path):
        p = tmp_path / 'data.json'
        JsonBackedDict(p, initial={'td': timedelta(0)})
        d2 = self._reload(p)
        assert d2['td'] == timedelta(0)

    def test_round_trip_negative(self, tmp_path):
        p = tmp_path / 'data.json'
        td = timedelta(hours=-5, minutes=-30)
        JsonBackedDict(p, initial={'td': td})
        d2 = self._reload(p)
        assert d2['td'] == td

    def test_round_trip_with_microseconds(self, tmp_path):
        p = tmp_path / 'data.json'
        td = timedelta(seconds=1, microseconds=500000)
        JsonBackedDict(p, initial={'td': td})
        d2 = self._reload(p)
        assert d2['td'] == td

    def test_in_list(self, tmp_path):
        p = tmp_path / 'data.json'
        tds = [timedelta(hours=i) for i in range(3)]
        JsonBackedDict(p, initial={'tds': tds})
        d2 = self._reload(p)
        assert d2['tds'] == tds


# ---------------------------------------------------------------------------
# 8. Atomicity
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_temp_file_in_same_directory(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        replaced_args: list[tuple[str, str]] = []

        original_replace = os.replace

        def mock_replace(src: str, dst: str) -> None:
            replaced_args.append((src, dst))
            original_replace(src, dst)

        with patch('json_backed_dict.core.os.replace', side_effect=mock_replace):
            d['k'] = 'v'

        assert len(replaced_args) == 1
        tmp_file, dest_file = replaced_args[0]
        assert Path(tmp_file).parent == Path(dest_file).parent == tmp_path

    def test_original_file_valid_if_replace_fails(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'original': True})

        def mock_replace(src: str, dst: str) -> None:
            raise OSError('disk full')

        with patch('json_backed_dict.core.os.replace', side_effect=mock_replace):
            with pytest.raises(OSError):
                d['new_key'] = 'new_value'

        assert load_raw(p) == {'original': True}

    def test_no_temp_file_left_on_failure(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        files_before = set(tmp_path.iterdir())

        def mock_replace(src: str, dst: str) -> None:
            raise OSError('disk full')

        with patch('json_backed_dict.core.os.replace', side_effect=mock_replace):
            with pytest.raises(OSError):
                d['k'] = 'v'

        assert set(tmp_path.iterdir()) == files_before


# ---------------------------------------------------------------------------
# 9. Round-trip (nested structures, unicode)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _reload(self, path: Path) -> JsonBackedDict:
        return JsonBackedDict(path)

    def test_nested_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        data = {'outer': {'inner': {'deep': 42}}}
        JsonBackedDict(p, initial=data)
        d2 = self._reload(p)
        assert d2['outer']['inner']['deep'] == 42

    def test_nested_list(self, tmp_path):
        p = tmp_path / 'data.json'
        data = {'items': [{'id': 1}, {'id': 2}]}
        JsonBackedDict(p, initial=data)
        d2 = self._reload(p)
        assert d2['items'] == [{'id': 1}, {'id': 2}]

    def test_unicode_values(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['greeting'] = '日本語テスト'
        d2 = self._reload(p)
        assert d2['greeting'] == '日本語テスト'
        raw_text = p.read_text(encoding='utf-8')
        assert '日本語テスト' in raw_text

    def test_all_scalar_types(self, tmp_path):
        p = tmp_path / 'data.json'
        data = {
            'string': 'hello',
            'integer': 42,
            'float': 3.14,
            'bool_true': True,
            'bool_false': False,
            'null': None,
        }
        JsonBackedDict(p, initial=data)
        d2 = self._reload(p)
        assert dict(d2) == data

    def test_bool_not_confused_with_int(self, tmp_path):
        p = tmp_path / 'data.json'
        JsonBackedDict(p, initial={'flag': True, 'num': 1})
        d2 = self._reload(p)
        assert d2['flag'] is True
        assert d2['num'] == 1

    def test_date_string_coerces_to_date_on_reload(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['v'] = date(2024, 1, 15)  # stored as "2024-01-15"
        d2 = self._reload(p)
        assert type(d2['v']) is date
        assert d2['v'] == date(2024, 1, 15)

    def test_datetime_string_coerces_to_datetime_on_reload(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['v'] = datetime(2024, 1, 15, 10, 30)
        d2 = self._reload(p)
        assert isinstance(d2['v'], datetime)
        assert d2['v'] == datetime(2024, 1, 15, 10, 30)

    def test_timedelta_string_coerces_to_timedelta_on_reload(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        d['v'] = timedelta(days=1, hours=2)  # stored as "1d2h"
        d2 = self._reload(p)
        assert isinstance(d2['v'], timedelta)
        assert d2['v'] == timedelta(days=1, hours=2)


# ---------------------------------------------------------------------------
# 10. Nested mutations
# ---------------------------------------------------------------------------


class TestNestedMutations:
    def _reload(self, path: Path) -> JsonBackedDict:
        return JsonBackedDict(path)

    def test_nested_dict_setitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'cfg': {'timeout': 10}})
        d['cfg']['timeout'] = 30
        d2 = self._reload(p)
        assert d2['cfg']['timeout'] == 30

    def test_nested_dict_delitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'cfg': {'a': 1, 'b': 2}})
        del d['cfg']['a']
        d2 = self._reload(p)
        assert 'a' not in d2['cfg']
        assert d2['cfg']['b'] == 2

    def test_deeply_nested_setitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': {'b': {'c': 1}}})
        d['a']['b']['c'] = 99
        d2 = self._reload(p)
        assert d2['a']['b']['c'] == 99

    def test_nested_list_append_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'items': [1, 2]})
        d['items'].append(3)
        d2 = self._reload(p)
        assert d2['items'] == [1, 2, 3]

    def test_nested_list_setitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'items': [1, 2, 3]})
        d['items'][0] = 99
        d2 = self._reload(p)
        assert d2['items'] == [99, 2, 3]

    def test_nested_list_delitem_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'items': [1, 2, 3]})
        del d['items'][1]
        d2 = self._reload(p)
        assert d2['items'] == [1, 3]

    def test_nested_list_extend_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'items': [1]})
        d['items'].extend([2, 3])
        d2 = self._reload(p)
        assert d2['items'] == [1, 2, 3]

    def test_assign_proxy_as_value(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'src': {'x': 1}, 'dst': {}})
        d['dst'] = d['src']
        d2 = self._reload(p)
        assert d2['dst'] == {'x': 1}

    def test_get_returns_proxy_for_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'cfg': {'k': 'v'}})
        proxy = d.get('cfg')
        proxy['k'] = 'changed'
        d2 = self._reload(p)
        assert d2['cfg']['k'] == 'changed'

    def test_nested_dict_update_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'cfg': {'a': 1}})
        d['cfg'].update({'b': 2})
        d2 = self._reload(p)
        assert d2['cfg'] == {'a': 1, 'b': 2}

    def test_nested_list_sort_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'nums': [3, 1, 2]})
        d['nums'].sort()
        d2 = self._reload(p)
        assert d2['nums'] == [1, 2, 3]

    def test_nested_list_reverse_persists(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'nums': [1, 2, 3]})
        d['nums'].reverse()
        d2 = self._reload(p)
        assert d2['nums'] == [3, 2, 1]


# ---------------------------------------------------------------------------
# 11. Additional edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    def test_pop_missing_with_default_no_file_write(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        mtime_before = p.stat().st_mtime_ns
        result = d.pop('missing', 'fallback')
        assert result == 'fallback'
        assert p.stat().st_mtime_ns == mtime_before

    def test_corrupted_json_raises_value_error_with_path(self, tmp_path):
        p = tmp_path / 'data.json'
        p.write_bytes(b'{not valid json')
        with pytest.raises(ValueError, match=str(p)):
            JsonBackedDict(p)


# ---------------------------------------------------------------------------
# 12. values() / items() are re-iterable (not single-use generators)
# ---------------------------------------------------------------------------


class TestValuesItemsReIterable:
    def test_values_can_be_iterated_twice(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1, 'b': 2})
        vals = d.values()
        assert list(vals) == list(vals)

    def test_items_can_be_iterated_twice(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        items = d.items()
        assert list(items) == list(items)

    def test_proxy_values_can_be_iterated_twice(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'cfg': {'x': 1, 'y': 2}})
        proxy = d['cfg']
        vals = proxy.values()
        assert list(vals) == list(vals)

    def test_proxy_items_can_be_iterated_twice(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'cfg': {'x': 1}})
        proxy = d['cfg']
        items = proxy.items()
        assert list(items) == list(items)


# ---------------------------------------------------------------------------
# 13. Double-close regression: write failure raises original error
# ---------------------------------------------------------------------------


class TestAtomicWriteDoubleClose:
    def test_write_failure_raises_original_error_not_bad_fd(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)

        original_fdopen = os.fdopen

        def mock_fdopen(fd, mode):
            f = original_fdopen(fd, mode)

            class FailingFile:
                def write(self, data):
                    raise OSError('simulated write failure')

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    f.close()
                    return False

            return FailingFile()

        with patch('json_backed_dict.core.os.fdopen', side_effect=mock_fdopen):
            with pytest.raises(OSError, match='simulated write failure'):
                d['k'] = 'v'


# ---------------------------------------------------------------------------
# 14. update() with empty input does not write
# ---------------------------------------------------------------------------


class TestUpdateNoWriteOnEmpty:
    def test_update_empty_dict_no_write(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        mtime_before = p.stat().st_mtime_ns
        d.update({})
        assert p.stat().st_mtime_ns == mtime_before

    def test_update_no_args_no_write(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        mtime_before = p.stat().st_mtime_ns
        d.update()
        assert p.stat().st_mtime_ns == mtime_before


# ---------------------------------------------------------------------------
# 15. |= operator (in-place merge)
# ---------------------------------------------------------------------------


class TestIorOperator:
    def test_ior_persists_to_file(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        d |= {'b': 2}
        assert load_raw(p) == {'a': 1, 'b': 2}

    def test_ior_updates_in_memory(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        d |= {'b': 2}
        assert dict(d) == {'a': 1, 'b': 2}

    def test_ior_returns_self(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        result = d.__ior__({'b': 2})
        assert result is d

    def test_ior_validates_types(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        with pytest.raises(TypeError):
            d |= {'b': object()}

    def test_ior_validates_keys(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        with pytest.raises(TypeError):
            d |= {1: 'bad_key'}  # type: ignore[operator]

    def test_ior_atomic_on_validation_failure(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        with pytest.raises(TypeError):
            d |= {'good': 'ok', 'bad': object()}
        # original content unchanged
        assert dict(d) == {'a': 1}
        assert load_raw(p) == {'a': 1}


# ---------------------------------------------------------------------------
# 16. | operator (non-mutating merge)
# ---------------------------------------------------------------------------


class TestOrOperator:
    def test_or_returns_plain_dict(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        result = d | {'b': 2}
        assert type(result) is dict

    def test_or_merged_contents(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'a': 1})
        result = d | {'b': 2}
        assert result == {'a': 1, 'b': 2}

    def test_or_does_not_mutate_original(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'a': 1})
        _ = d | {'b': 2}
        assert dict(d) == {'a': 1}
        assert load_raw(p) == {'a': 1}

    def test_ror_plain_dict_or_jbd(self, tmp_path):
        d = JsonBackedDict(tmp_path / 'data.json', initial={'b': 2})
        result = {'a': 1} | d
        assert type(result) is dict
        assert result == {'a': 1, 'b': 2}

    def test_or_does_not_leak_nested_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'nested': {'key': 'val'}})
        result = d | {'other': 1}
        result['nested']['key'] = 'hacked'
        assert d['nested']['key'] == 'val'
        assert load_raw(p)['nested']['key'] == 'val'

    def test_or_does_not_leak_nested_list(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'items': [1, 2]})
        result = d | {'other': 1}
        result['items'].append(99)
        assert d['items'] == [1, 2]
        assert load_raw(p)['items'] == [1, 2]

    def test_ror_does_not_leak_nested_dict(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'b': 2, 'nested': {'key': 'val'}})
        result: dict[str, Any] = cast(dict[str, Any], {'a': 1} | d)
        result['nested']['key'] = 'hacked'
        assert d['nested']['key'] == 'val'
        assert load_raw(p)['nested']['key'] == 'val'

    def test_ror_does_not_leak_nested_list(self, tmp_path):
        p = tmp_path / 'data.json'
        d = JsonBackedDict(p, initial={'b': 2, 'items': [1, 2]})
        result: dict[str, Any] = cast(dict[str, Any], {'a': 1} | d)
        result['items'].append(99)
        assert d['items'] == [1, 2]
        assert load_raw(p)['items'] == [1, 2]


class TestThreadSafety:
    def test_concurrent_independent_writes(self, tmp_path):
        import threading

        d = JsonBackedDict(tmp_path / 'data.json')
        barrier = threading.Barrier(10)
        errors = []

        def worker(i):
            barrier.wait()
            try:
                d[f'key_{i}'] = i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(d) == 10
        for i in range(10):
            assert d[f'key_{i}'] == i

    def test_concurrent_reads_and_writes(self, tmp_path):
        import threading

        d = JsonBackedDict(tmp_path / 'data.json', initial={'x': 0})
        stop = threading.Event()
        errors = []

        def writer():
            for i in range(100):
                try:
                    d['x'] = i
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop.is_set():
                try:
                    _ = d.get('x')
                except Exception as e:
                    errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(5)]
        writer_thread = threading.Thread(target=writer)
        for r in readers:
            r.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for r in readers:
            r.join()
        assert not errors

    def test_concurrent_proxy_mutations(self, tmp_path):
        import threading

        d = JsonBackedDict(tmp_path / 'data.json', initial={'items': []})
        barrier = threading.Barrier(10)
        errors = []

        def worker(i):
            barrier.wait()
            try:
                d['items'].append(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(d['items']) == 10

    def test_disk_matches_memory_after_concurrent_writes(self, tmp_path):
        import threading

        p = tmp_path / 'data.json'
        d = JsonBackedDict(p)
        barrier = threading.Barrier(10)

        def worker(i):
            barrier.wait()
            d[f'k{i}'] = i

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reloaded = JsonBackedDict(p)
        assert dict(reloaded) == dict(d)
