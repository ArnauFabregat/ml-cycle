import copy
import pandas
from typing import List, Dict

# Esto se puede limpiar un poco porque está en functions
builtin_agg_functions = ['min', 'max', 'mean', 'sum', 'first', 'count']


def _field_range(s: pandas.Series) -> int | float:
    """Auxiliary fuction to perfom the range between the values of a pandas Series

    Args:
        s: input pandas series.

    Returns:
        range in the pandas Series.
    """
    return max(s) - min(s)


def _join_s(s: pandas.Series) -> str:
    """Auxiliary fuction to perfom a union of the strings of a pandas Series

    Args:
        s: input pandas series.

    Returns:
        union of Strings.
    """
    return ' '.join(s)


def _intersection(lst1: List, lst2: List) -> List:
    """Auxiliary function to perform intersections between two lists.

    Args:
        lst1: first list to intersect.
        lst2: second list to intersect.

    Returns:
        resulting list with intersection.
    """
    return list(set(lst1) & set(lst2))


def _difference(lst1: List, lst2: List) -> List:
    """ Auxiliary function to perform differences between two lists.

    Args:
        lst1: first list to make the difference.
        lst2: second list to make the difference.

    Returns:
        resulting difference between list.
    """
    return list(set(lst1).difference(lst2))


def _validate_functions_to_apply_from_dict(aggregation_dict: Dict) -> Dict:
    """There are a list of built-in functions available within pandas aggregations. However, there might be other custom
    business needs not covered by pandas aggregation functions.
    'validate_functions_to_apply_from_dict' maps those custom functions from their name to the actual stored in the
    object internal_dict. In order to add new functions to config.json, they may be stored in the internal_dict.

    Args:
        aggregation_dict: input data dictionary, may contain as value strings or list of strings.

    Returns:
        output data dictionary with the aggregations defined by users.
    """
    internal_dict = {
        '_field_range': _field_range,
        '_join_s': _join_s,
    }
    agg_dict_copy = copy.deepcopy(aggregation_dict)
    for key, value in aggregation_dict.items():
        if isinstance(value, list):
            aux = []
            builtin_function_list = _intersection(value, builtin_agg_functions)
            custom_function_list = _difference(value, builtin_agg_functions)
            for custom_function in custom_function_list:
                aux.append(internal_dict[custom_function])
            composed_list = builtin_function_list + aux
            agg_dict_copy.upandasate({key: list(set(composed_list))})
        elif value not in builtin_agg_functions:
            agg_dict_copy.upandasate({key: internal_dict[value]})
    return agg_dict_copy


def group_by_map(df: pandas.DataFrame, group_by_column: List, aggregation_dict: Dict) -> pandas.DataFrame:
    """Groups by a list of column/s and computes an aggregation function (one or more) per column suggested in the
    aggregation_map dict. Then, operations and column names are concatenated (except if the operation is first, keeping
    the actual column name).

    NOTE:
        Aggregation functions **enabled for now**: *mean*, *sum*, *count*,
        *std*, *var*, *first*, *last*, *min*, *max*, *sem*, *nunique*.

        **Not enabled yet**: nth(n) -> requires an input, describe -> create sub-indexes.

    Args:
        df: input pandas Dataframe.
        group_by_column: list of column/s name/s to group by.
        aggregation_dict: python dict `{"column":["functions_to_agg_with",...]}`.

    Returns:
        output df with aggregations done.

    Examples:
        >>> import pandas as pd
        >>> from mlcycle.feature_engineering.Aggregate import group_by_map
        >>> header = ['agg_column','num_column_1', 'num_column_2',
                      'cat_column_1', 'cat_column_2', 'time_column']
        >>> data = [
                ['a', 0, 0.0, 'a', 2, '01/01/2001'],
                ['a', 1, 1.0, 'a', 3, '02/01/2001'],
                ['b', 14, 1.0, 'b', 1, '02/01/2001'],
                ['b',-3, 1.5, 'a', 1, '02/01/2001'],
                ['c', 21, 1.0, 'b', 1, '03/01/2001'],
                ['c',1, 2.0, 'a', 1, '05/01/2001'],
            ]
        >>> agg_column = 'agg_column'
        >>> aggregation_features = {
                'cat_column_1' : 'first',
                'cat_column_2' : 'max',
                'time_column' : 'first',
                'num_column_1' : ['mean', 'min', 'max'],
                'num_column_2' : ['mean', 'min', 'max'],
            }
        >>> df = pd.DataFrame(data, columns=header)
        >>> group_by_map(df, ['agg_column'], aggregation_features)
    """
    final_dict = _validate_functions_to_apply_from_dict(aggregation_dict)
    multiindex_df = df.groupby(group_by_column).agg(final_dict)
    multiindex_df.columns = [x[0] if x[1] in 'first' else '_'.join(map(str, x)) for x in multiindex_df.columns]
    return multiindex_df
