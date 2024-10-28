# cache.py

import numpy as np
from collections.abc import MutableMapping
from typing import Optional, Union, List, Dict, Any
import uuid
import multiprocessing
import os
import pickle


class DataArray:
    """
    Represents a data array in the cache. It can be a directly stored numpy array,
    or an aggregation of other arrays in the cache. It can also store the transformer
    used to create the data array for inverse transformations.
    """

    def __init__(self, np_array: Optional[np.ndarray] = None, type: Optional[str] = None,
                 sources_id: Optional[Union[str, List[str]]] = None, rows: Optional[np.ndarray] = None,
                 transformer: Optional[Any] = None):
        self.np_array = np_array  # Directly stored numpy array
        self.type = type  # Type of aggregation: 'concat', 'augment', etc.
        self.sources_id = sources_id  # List of source IDs for aggregation
        self.rows = rows  # Row indices for filtering
        self.transformer = transformer  # Transformer used to create this data array
        self.indices = [] if np_array is None else np.arange(np_array.shape[0])

    def get_data(self, cache: 'DataCache', aggregation='concat') -> np.ndarray:
        """
        Retrieve the data associated with this DataArray.

        :param cache: The cache instance to retrieve data from.
        :return: The numpy array representing the data.
        """
        if self.np_array is not None:
            return self.np_array
        elif self.type in ('undefined', 'augment', 'concat', 'union', 'transpose_union'):
            if self.type == 'undefined':
                return self._aggregate_data(cache, aggregation)
            else:
                return self._aggregate_data(cache, self.type)
        # elif self.type == "filter":
        #     return cache.get(self.sources_id)[self.rows]
        else:
            raise ValueError("No data found in DataArray.")

    def _aggregate_data(self, cache: 'DataCache', aggregation) -> np.ndarray:
        """
        Aggregate data from source arrays based on the aggregation type.

        :param cache: The cache instance to retrieve data from.
        :return: The aggregated numpy array.
        """
        arrays = [cache.get(src_id) for src_id in self.sources_id]
        aggregation_type = aggregation
        if aggregation_type == 'augment':
            return np.vstack(arrays)
        elif aggregation_type == 'concat':
            return np.concatenate(arrays, axis=1)
        elif aggregation_type == 'union':
            return np.stack(arrays, axis=-1)
        elif aggregation_type == 'transpose_union':
            return np.stack(arrays, axis=1)
        else:
            raise ValueError(f"Unknown aggregation type: {self.type}")

    def get_transformer(self) -> Optional[Any]:
        """
        Retrieve the transformer associated with this DataArray.

        :return: The transformer object or None if not available.
        """
        return self.transformer


class DataCache(MutableMapping):
    """
    Class for caching data arrays. Supports storing and retrieving data arrays,
    including aggregated data arrays that are computed on-the-fly from source arrays.
    It also stores transformers used for data transformations to enable inverse transformations.
    """

    _instance = None
    transformers_dir = 'transformers'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
            cls._instance._init_cache()
        return cls._instance

    def _init_cache(self):
        # manager = multiprocessing.Manager()
        self._cache = dict()#manager.dict()

    def clear(self) -> None:
        """
        Clear the cache.
        """
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        :param key: The key to check.
        :return: True if the key exists, False otherwise.
        """
        return key in self._cache

    def get(self, key: str, aggregation='concat') -> np.ndarray:
        if not self.exists(key):
            raise KeyError(f"Key '{key}' not found in cache.")
        data_array = self._cache[key]
        return data_array.get_data(self, aggregation=aggregation)

    def get_transformer(self, key: str) -> Optional[Any]:
        """
        Get the transformer associated with a key.

        :param key: The key to retrieve the transformer for.
        :return: The transformer object or None if not available.
        :raises KeyError: If the key is not found.
        """
        if not self.exists(key):
            raise KeyError(f"Key '{key}' not found in cache.")
        return self._cache[key].get_transformer()

    def set(self, key: str, value: np.ndarray) -> None:
        """
        Set the data for a key.

        :param key: The key to set.
        :param value: The numpy array to store.
        """
        self._cache[key] = DataArray(np_array=value)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self.set(key, value)

    # def set_filtered(self, key: str, src_key: str, row_indices: np.ndarray) -> None:
    #     """
    #     Set a filtered data array in the cache.

    #     :param key: The key to set.
    #     :param src_key: The source data key.
    #     :param row_indices: The indices to filter.
    #     """
    #     if not self.exists(src_key):
    #         raise ValueError(f"No data found in cache with key '{src_key}'.")
    #     self._cache[key] = DataArray(type="filter", sources_id=src_key, rows=row_indices)

    def set_transformed(self, key: str, value: np.ndarray, src_key: str, transformer: Optional[Any] = None) -> None:
        """
        Set a transformed data array in the cache.

        :param key: The key to set.
        :param value: The transformed numpy array.
        :param src_key: The source data key.
        :param transformer: The transformer used to create the transformed data.
        """
        if not self.exists(src_key):
            raise ValueError(f"No data found in cache with key '{src_key}'.")
        self._cache[key] = DataArray(np_array=value, sources_id=src_key, transformer=transformer)


    def set_aggregation(self, key: str, sources: List[str], agg_type: str = 'undefined', transformers: Optional[List[Any]] = None) -> None:
        """
        Set an aggregated data array in the cache.

        :param key: The key to set.
        :param sources: List of source data keys.
        :param agg_type: The type of aggregation ('concat', 'augment', etc.).
        :param transformers: List of transformers used for each source.
        """
        for src in sources:
            if not self.exists(src):
                raise ValueError(f"No data found in cache with key '{src}'.")
        self._cache[key] = DataArray(type=agg_type, sources_id=sources, transformer=transformers)

    def __delitem__(self, key: str) -> None:
        del self._cache[key]

    def __iter__(self):
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def revert(self, key: str) -> Optional[np.ndarray]:
        """
        Revert to the source data of a key.

        :param key: The key to revert.
        :return: The source data as a numpy array.
        :raises KeyError: If the key is not found.
        """
        if not self.exists(key):
            raise KeyError(f"Key '{key}' not found in cache.")
        data_array = self._cache[key]
        if data_array.np_array is not None:
            return data_array.np_array
        elif data_array.sources_id:
            if isinstance(data_array.sources_id, str):
                return self.get(data_array.sources_id)
            elif isinstance(data_array.sources_id, list):
                return [self.get(src_id) for src_id in data_array.sources_id]
        return None

    def deep_copy(self, key: str, id_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Create a deep copy of the data associated with the given key.
        Returns the new key of the copied data.

        :param key: The key of the data to copy.
        :param id_mapping: A dictionary mapping old IDs to new IDs, to handle cycles.
        :return: The new key of the copied data.
        """

        if id_mapping is None:
            id_mapping = {}
        if key in id_mapping:
            # Already copied
            return id_mapping[key]
        if key not in self._cache:
            raise KeyError(f"Key '{key}' not found in cache.")
        data_array = self._cache[key]
        # Generate new key
        new_key = uuid.uuid4().hex[:8]
        id_mapping[key] = new_key
        # Copy data_array
        if data_array.np_array is not None:
            # Directly stored numpy array
            new_np_array = np.copy(data_array.np_array)
            new_data_array = DataArray(np_array=new_np_array)
            self._cache[new_key] = new_data_array
        elif data_array.type in ('augment', 'concat', 'union', 'transpose_union'):
            # Aggregated data, recursively copy sources
            new_sources_id = []
            for src_id in data_array.sources_id:
                new_src_id = self.deep_copy(src_id, id_mapping)
                new_sources_id.append(new_src_id)
            new_data_array = DataArray(type=data_array.type, sources_id=new_sources_id)
            self._cache[new_key] = new_data_array
        # elif data_array.type == 'filter':
        #     # Filtered data, recursively copy source
        #     new_src_id = self.deep_copy(data_array.sources_id, id_mapping)
        #     new_rows = np.copy(data_array.rows)
        #     new_data_array = DataArray(type='filter', sources_id=new_src_id, rows=new_rows)
        #     self._cache[new_key] = new_data_array
        else:
            raise ValueError("Unknown DataArray type during deep copy.")
        return new_key
    
    def _save_transformer(self, data_id, transformer):
        os.makedirs(self.transformers_dir, exist_ok=True)
        transformer_path = os.path.join(self.transformers_dir, f"{data_id}_transformer.pkl")
        with open(transformer_path, 'wb') as f:
            pickle.dump(transformer, f)

    def _load_transformer(self, data_id):
        transformer_path = os.path.join(self.transformers_dir, f"{data_id}_transformer.pkl")
        if os.path.exists(transformer_path):
            with open(transformer_path, 'rb') as f:
                transformer = pickle.load(f)
            return transformer
        else:
            return None


# Create a shared cache instance
# cache = DataCache()


Let's illustrate. 
If X = [[1, 1], [2, 2], [3, 3]]. 

I apply a sample_augmentation [None, Add1, Div2]

I obtain X_augmented = [ [[1, 1], [2, 2], [3, 3]], [[2, 2], [3, 3], [4, 4]], [[0.5, 0.5], [1, 1], [1.5, 1.5]]]

What I want is to be able to do either
getData(filter=[true, false, false]) or getData(filter=[1]) and retrieve the first index of all augmented set and the complementary

filteredX = getData(filter=[1, 2])
filteredX:
[[1, 1], [2, 2], [2, 2], [3, 3], [0.5, 0.5], [1, 1]]

If X is transformed with a feature augmentation[None, Add3], then X will become

X = [ [ [ [1,1],[4,4] ], [ [2,2],[5,5] ], [ [3,3],[6,6] ] ], [ [ [2,2],[5,5] ], [3,3],[6,6], [4,4],[7,7]], [ [ [0.5,0.5],[3.5,3.5] ], [ [1,1],[4,4] ], [ [1.5,1.5],[4.5,4.5] ] ] ]

Then I can ask:
    filteredX= getData(filter=[1, 2])
    filteredX:
    [ [ [1,1],[4,4] ], [ [2,2],[5,5] ], [ [2,2],[5,5] ], [ [3,3],[6,6] ], [ [0.5,0.5],[3.5,3.5] ], [ [1,1],[4,4] ] ]