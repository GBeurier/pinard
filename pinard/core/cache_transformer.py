import hashlib
import numpy as np
from joblib import Parallel, delayed, dump, load
from sklearn.base import TransformerMixin


class Cache_Transformer():
    def __init__(self, pipeline, cache):
        self.pipeline = pipeline
        self.cache = cache

    def get_id(self, src_id, single_transformer):
        key_str = src_id + '_' + single_transformer.__class__.__name__ #TODO replace with custom str from config
        return hashlib.md5(key_str.encode()).hexdigest()[0:8]

    def execute(self, id):
        # get X
        # process pipeline
                
        # """Execute the pipeline on input data X."""
        # # Initial data ID and transformation chain
        # data_id = 'o_'
        # cache.add(data_id, X)

        # x_transformed, x_transformed_id = self._process_pipeline(X, self.pipeline, data_id)

        # return x_transformed, x_transformed_id
        return None, None


    def _process_pipeline(self, X, pipeline, data_id):
        current_data = X
        src_id = data_id
        for step in pipeline:
            if isinstance(step, dict):
                if 'sample_augmentation' in step or "+" in step:
                    current_data, src_id = self._handle_sample_augmentation(current_data, step['sample_augmentation'], src_id)
                elif 'feature_augmentation' in step or "*" in step:
                    current_data, src_id = self._handle_feature_augmentation(current_data, step['feature_augmentation'], src_id)
            else:
                current_data, src_id = self._apply_transformer(current_data, step, data_id, step)
        return current_data, src_id

    def _apply_transformer(self, X, transformer, data_id):
        cache_key = self.get_cache_key(data_id, transformer)
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            if isinstance(X, list):
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(
                        lambda x: transformer.fit_transform(x), X))
            else:
                results = transformer.fit_transform(X)
            # Cache the result
            self.cache[cache_key] = results
            return results

    def _handle_sample_augmentation(self, X, augmentations, data_id, transformations):
        """Handle sample augmentation."""
        datasets = []
        new_transformations = transformations + ['sample_augmentation']
        with ThreadPoolExecutor() as executor:
            futures = []
            for aug in augmentations:
                aug_transformations = new_transformations + [aug]
                cache_key = self.get_cache_key(data_id, aug_transformations)
                if cache_key in self.cache:
                    datasets.append(self.cache[cache_key])
                else:
                    # Apply augmentation
                    if aug is None:
                        aug_data = X
                    else:
                        aug_data = aug.fit_transform(X)
                    self.cache[cache_key] = aug_data
                    datasets.append(aug_data)
        return datasets

    def _handle_feature_augmentation(self, datasets, augmentations, data_id, transformations):
        """Handle feature augmentation."""
        augmented_datasets = []
        new_transformations = transformations + ['feature_augmentation']
        with ThreadPoolExecutor() as executor:
            for dataset in datasets:
                dataset_results = []
                for aug in augmentations:
                    aug_transformations = new_transformations + [aug]
                    cache_key = self.get_cache_key(data_id, aug_transformations)
                    if cache_key in self.cache:
                        dataset_results.append(self.cache[cache_key])
                    else:
                        # Apply augmentation
                        if aug is None:
                            aug_data = dataset
                        else:
                            aug_data = aug.fit_transform(dataset)
                        self.cache[cache_key] = aug_data
                        dataset_results.append(aug_data)
                # Combine feature augmentations
                augmented_datasets.append(dataset_results)
        return augmented_datasets

    def get_data(self, aggregation='concat', filter_indices=None):
        """Retrieve data from cache with specified aggregation and filter."""
        # Flatten the cached datasets
        datasets = []
        for key in self.cache:
            datasets.append(self.cache[key])
        # Apply filter
        if filter_indices is not None:
            datasets = [data[filter_indices] for data in datasets]
        # Apply aggregation
        if aggregation == 'concat':
            return np.vstack([data.reshape(data.shape[0], -1) for data in datasets])
        elif aggregation == 'interlaced':
            interlaced_data = np.hstack([data.reshape(data.shape[0], -1) for data in datasets])
            return interlaced_data
        elif aggregation == 'union':
            union_data = np.stack(datasets, axis=2)
            return union_data
        elif aggregation == 'transpose_union':
            transpose_union_data = np.stack(datasets, axis=1)
            return transpose_union_data
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
