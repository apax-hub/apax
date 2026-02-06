from apax.utils.helpers import update_nested_dictionary


def test_update_nested_dictionary():
    d1 = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    d2 = {"b": {"c": 4, "f": 5}}
    expected_dict = {"a": 1, "b": {"c": 4, "d": {"e": 3}, "f": 5}}
    updated_dict = update_nested_dictionary(d1, d2)
    assert updated_dict == expected_dict
