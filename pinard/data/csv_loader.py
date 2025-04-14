import csv
import io
import numpy as np
import pandas as pd
import gzip
import zipfile
from pathlib import Path
import re


# =============================================================================
# Utility: Check if a string can be converted to float, given a decimal separator
# =============================================================================
def _can_be_float(value, decimal_sep):
    """Check if a string can be converted to a float."""
    if not isinstance(value, str):
        return False  # Only strings should be checked

    value = value.strip()
    if not value:
        return False

    try:
        # If there's 'e' or 'E', handle scientific notation
        if 'e' in value.lower() or 'E' in value:
            float(value)
            return True

        # Replace decimal separator if needed
        if decimal_sep == '.':
            float(value)
        else:
            float(value.replace(decimal_sep, '.', 1))
        return True
    except ValueError:
        return False


# =============================================================================
# Utility: Strip all quotes from the file content
# =============================================================================
def _strip_all_quotes(content):
    """
    Removes *all* single-quote and double-quote characters from the string.
    """
    return content.replace('"', '').replace("'", "")


# =============================================================================
# Detect delimiter (unless specified by user)
# =============================================================================
def _detect_delimiter(lines, possible_delimiters=None):
    """
    Attempt to detect the delimiter by looking at the consistency of
    the number of columns. Return the best candidate or None if no good guess.
    """
    if possible_delimiters is None:
        possible_delimiters = [';', ',', '\t', '|', ' ']

    best_delim = None
    max_consistent_cols = -1
    most_cols_at_max_consistency = 0
    
    # Join lines so csv.reader sees them as input
    content_for_test = "".join(lines)

    for delim_candidate in possible_delimiters:
        try:
            reader = csv.reader(io.StringIO(content_for_test), delimiter=delim_candidate)
            cols_counts = [len(row) for row in reader if row]

            if not cols_counts:
                continue

            # The number of columns that appears the most
            most_frequent_cols = max(set(cols_counts), key=cols_counts.count)
            # How many lines have that number of columns
            consistency = sum(1 for count in cols_counts if count == most_frequent_cols)

            # Choose the delimiter that maximizes the consistency, then the number of columns
            if consistency > max_consistent_cols:
                max_consistent_cols = consistency
                most_cols_at_max_consistency = most_frequent_cols
                best_delim = delim_candidate
            elif consistency == max_consistent_cols:
                # If same consistency, prefer the one with more columns
                if most_frequent_cols > most_cols_at_max_consistency:
                    most_cols_at_max_consistency = most_frequent_cols
                    best_delim = delim_candidate
        except Exception:
            continue  # ignore parse errors with this candidate

    return best_delim


# =============================================================================
# Detect decimal separator and header (unless specified by user)
# =============================================================================
def _detect_decimal_and_header(parsed_rows, data_type='x'):
    """
    Given a list of parsed_rows (already split by delimiter),
    try to determine the decimal separator and whether there's a header.

    Returns: (best_decimal_sep, best_has_header)
    """
    if not parsed_rows:
        return '.', False  # fallback

    # We'll guess it by looking at numeric vs. non-numeric content
    num_cols = len(parsed_rows[0])
    if num_cols == 0:
        return '.', False  # fallback

    best_decimal_sep = '.'
    best_has_header = False
    max_numeric_score = -1.0

    for decimal_sep in ['.', ',']:
        for has_header_option in [False, True]:
            first_data_row_index = 1 if has_header_option else 0
            if len(parsed_rows) <= first_data_row_index:
                # no data rows to evaluate
                current_score = 0.0
            else:
                data_rows = parsed_rows[first_data_row_index:]
                numeric_cells = 0
                total_cells = 0
                
                for row in data_rows:
                    # We only consider rows with at least close to the expected columns
                    if abs(len(row) - num_cols) <= 1:
                        for val in row:
                            total_cells += 1
                            if _can_be_float(val, decimal_sep):
                                numeric_cells += 1

                current_score = numeric_cells / total_cells if total_cells else 0.0

            # If we declared there's a header but that row also looks numeric,
            # apply a small penalty
            if has_header_option and parsed_rows:
                header_row = parsed_rows[0]
                if len(header_row) == num_cols:
                    header_numeric_cells = sum(_can_be_float(cell, decimal_sep) for cell in header_row)
                    header_score = header_numeric_cells / len(header_row) if header_row else 0.0
                    if current_score > 0.5 and header_score >= current_score:
                        current_score *= 0.5

            if current_score > max_numeric_score + 1e-6:
                max_numeric_score = current_score
                best_decimal_sep = decimal_sep
                best_has_header = has_header_option
            elif abs(current_score - max_numeric_score) < 1e-6:
                # Tie-break: prefer '.' over ',' and prefer has_header=False over True
                if best_decimal_sep == ',' and decimal_sep == '.':
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option
                elif best_has_header and (not has_header_option):
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option

    return best_decimal_sep, best_has_header


# =============================================================================
# Main routine: Determine CSV parameters, skipping detection if user param is given
# =============================================================================
def _determine_csv_parameters(csv_content: str,
                              sample_lines=20,
                              data_type='x',
                              user_params=None):
    """
    Inspect the first few lines of a CSV content (str) to auto-detect delimiter,
    decimal separator, and header if they're not already specified in `user_params`.
    """
    if user_params is None:
        user_params = {}

    # Split the content into lines
    lines = []
    with io.StringIO(csv_content) as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            if line.strip():
                lines.append(line)

    if not lines:
        # no lines to parse
        return {
            'delimiter': user_params.get('delimiter', ','),
            'decimal_separator': user_params.get('decimal_separator', '.'),
            'has_header': user_params.get('has_header', False)
        }

    # 1) Delimiter detection
    if 'delimiter' in user_params:
        delimiter = user_params['delimiter']
    else:
        delimiter = _detect_delimiter(lines)

    if not delimiter:
        # fallback
        delimiter = ','

    # 2) Parse a small sample using the chosen delimiter to create parsed_rows
    sample_data = "".join(lines)
    parsed_rows_reader = csv.reader(io.StringIO(sample_data), delimiter=delimiter)
    parsed_rows = [row for row in parsed_rows_reader if any(cell.strip() for cell in row)]

    # 3) Detect decimal separator / header if not specified
    if 'decimal_separator' in user_params:
        decimal_sep = user_params['decimal_separator']
    else:
        decimal_sep, _ = _detect_decimal_and_header(parsed_rows, data_type=data_type)

    if 'has_header' in user_params:
        has_header = user_params['has_header']
    else:
        _, has_header = _detect_decimal_and_header(parsed_rows, data_type=data_type)

    return {
        'delimiter': delimiter,
        'decimal_separator': decimal_sep,
        'has_header': has_header
    }


# =============================================================================
# Main function: load_csv
# =============================================================================
def load_csv(path, na_policy='auto', data_type='x', categorical_mode='auto', **user_params):
    """
    Loads a CSV file, detects parameters (unless overridden in user_params),
    cleans data, and returns a NumPy float32 array plus a 'report' dictionary.

    Args:
        path (str or Path): Path to the CSV file (.csv, .gz, .zip).
        na_policy (str): 'remove' or 'abort' (or 'auto' which acts like 'remove').
        data_type (str): 'x' or 'y' (informational).
        categorical_mode (str): How to handle categorical columns:
            - 'auto': Automatically detect and convert string columns to numerical
            - 'preserve': Keep string columns as is (will be converted to NaN when converting to numpy)
            - 'none': Don't detect categorical columns at all
        **user_params: CSV parsing parameters provided by the user:
            - delimiter
            - decimal_separator
            - has_header
            ... any other valid pandas.read_csv parameters are also accepted.

    Returns:
        (numpy.ndarray | None, dict):
            - Cleaned numeric data as float32 or None if error.
            - A dictionary 'report' detailing the loading process, including categorical mappings.
    """
    if na_policy == 'auto':
        na_policy = 'remove'

    if na_policy not in ['remove', 'abort']:
        raise ValueError("Invalid NA policy - only 'remove' or 'abort' (or 'auto') are supported.")    
    if categorical_mode not in ['auto', 'preserve', 'none']:
        raise ValueError("Invalid categorical mode - only 'auto', 'preserve', or 'none' are supported.")

    report = {
        'file_path': str(path),
        'detection_params': None,
        'delimiter': None,  # For backward compatibility
        'decimal_separator': None,  # If needed
        'has_header': None,  # If needed
        'initial_shape': None,
        'final_shape': None,
        'na_handling': {
            'strategy': na_policy,
            'na_detected': False,
            'nb_removed_rows': 0,
            'removed_rows_indices': []
        },
        'categorical_info': {},  # Store category mappings
        'warnings': [],  # Store warnings about ambiguous detections
        'error': None
    }

    try:
        file_path = Path(path)
        if not file_path.exists():
            # The test expects either "n'existe pas" or "not exist" in the error
            raise FileNotFoundError(f"Le fichier n'existe pas: {path}")

        # --- 1) Read file content ---
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
        elif file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files_in_zip = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_files_in_zip:
                    raise ValueError(f"No .csv found in zip: {path}")
                if len(csv_files_in_zip) > 1:
                    print(f"Warning: multiple .csv found in {path}. Using {csv_files_in_zip[0]}")
                content = z.read(csv_files_in_zip[0]).decode('utf-8')
        else:
            # plain text read
            try:
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: failed reading {path} with UTF-8. Trying Latin-1.")
                with open(file_path, 'r', encoding='latin-1', newline='') as f:
                    content = f.read()

        if not content.strip():
            raise ValueError("File is empty or could not be read.")

        # --- 2) Determine CSV parameters (unless overridden) ---
        # NOTE: _strip_all_quotes was removed from here. Parsing now happens on raw content.
        detection_params = _determine_csv_parameters(
            csv_content=content, # Use original content
            sample_lines=50,
            data_type=data_type,
            user_params=user_params
        )

        # Merge detection with user_params
        delimiter = user_params.get('delimiter', detection_params['delimiter'])
        decimal_sep = user_params.get('decimal_separator', detection_params['decimal_separator'])
        has_header = user_params.get('has_header', detection_params['has_header'])

        # Update report for backward compatibility
        report['detection_params'] = {
            'delimiter': delimiter,
            'decimal_separator': decimal_sep,
            'has_header': has_header
        }
        report['delimiter'] = delimiter
        report['decimal_separator'] = decimal_sep
        report['has_header'] = has_header

        # --- 4) Load with pandas.read_csv using the determined parameters ---
        # Use io.StringIO(content) which contains the original, unstripped content
        read_csv_kwargs = {
            'sep': delimiter,
            'decimal': decimal_sep,
            'header': 0 if has_header else None,
            'na_filter': True,
            'na_values': ['NA', 'N/A', ''],
            'keep_default_na': True,
            'engine': 'python', # Start with python engine for better error reporting/flexibility
            'skip_blank_lines': True,
            'quoting': csv.QUOTE_MINIMAL, # Default pandas behavior, explicit here
        }

        # Add user-provided read_csv args (besides the three we handled)
        for k, v in user_params.items():
            if k not in ['delimiter', 'decimal_separator', 'has_header']:
                read_csv_kwargs[k] = v

        try:
            # Pass the original content to StringIO
            data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
        except Exception as e1:
            print(f"Warning: read_csv with engine='python' failed: {e1}")
            try:
                read_csv_kwargs['engine'] = 'c'
                data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
            except Exception as e2:
                msg = f"Could not parse CSV. Python engine error: {e1} | C engine error: {e2}"
                report['error'] = msg
                return None, report        
        report['initial_shape'] = data.shape

        # ---> FIX: Ensure column names are strings <--- 
        data.columns = data.columns.astype(str)

        # --- 5) Handle string/categorical columns & Identify intended numeric ---
        intended_numeric_columns = []
        columns_to_factorize = []
        categorical_mappings = {}
        report['categorical_info'] = {} # Reset here

        for col in data.columns: # Already strings
            col_dtype = data[col].dtype
            is_object = pd.api.types.is_object_dtype(col_dtype)
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype)

            if is_numeric:
                intended_numeric_columns.append(col)
                continue

            if is_object and categorical_mode != 'none':
                unique_values_series = data[col].dropna()
                if unique_values_series.empty:
                    intended_numeric_columns.append(col) # Treat empty object columns as numeric (will be NaN)
                    continue

                # Check if object column looks numeric
                # Avoid checking tiny columns
                is_mostly_numeric = False
                if len(unique_values_series) > 5: 
                    numeric_check = pd.to_numeric(unique_values_series, errors='coerce')
                    # If less than 10% are NaN after conversion, assume it was meant to be numeric.
                    is_mostly_numeric = not numeric_check.isna().all() and (numeric_check.isna().sum() / len(numeric_check) < 0.1)

                if is_mostly_numeric:
                     intended_numeric_columns.append(col)
                     print(f"Info: Column '{col}' is object type but appears mostly numeric. Treating as numeric.")
                     continue # Treat as numeric                # If it's object, not empty, not mostly numeric, check categorical modes
                if categorical_mode == 'auto':
                    unique_values = unique_values_series.unique()
                    # Heuristic: Check unique count relative to total rows
                    max_unique_allowed = max(5, int(data.shape[0] * 0.8)) 
                    if len(unique_values) > 0 and len(unique_values) <= max_unique_allowed:
                        # Issue warning if header looks numeric
                        # Simple check if header is numeric (handles integers and floats)
                        if col.replace('.', '', 1).lstrip('-').isdigit(): 
                             warning_msg = f"Column '{col}' detected as categorical but has a numeric header. Check data integrity."
                             if warning_msg not in report['warnings']: report['warnings'].append(warning_msg)
                        columns_to_factorize.append(col)
                    else:
                        # Auto mode failed heuristic, treat as non-numeric (will become NaN)
                        # Don't add to intended_numeric_columns, let it become NaN silently in preserve/none style
                        print(f"Info: Column '{col}' has too many unique values ({len(unique_values)}) for auto categorical detection. Treating as non-numeric.")

                # Preserve and none modes should both exclude string columns from intended_numeric_columns
                # so they aren't checked for NAs and then dropped
            
            # If categorical_mode is 'none', or if it's 'object' but not numeric-like and not factorized by 'auto',
            # it will fall through here. We don't add it to intended_numeric_columns.
            # It will be coerced to NaN in the next step.

            # Handle other non-numeric, non-object types (like bool?) - treat as numeric for now
            elif not is_object and not is_numeric:
                 intended_numeric_columns.append(col)


        # Factorize columns marked for it (only in 'auto' mode)
        for col in columns_to_factorize:
             # Ensure consistency: Convert column to string before factorizing to handle mixed types gracefully
             codes, categories = pd.factorize(data[col].astype(str)) 
             # Handle potential -1 codes from factorizing NaN values if necessary (though factorize usually handles them)
             # codes = np.where(codes == -1, np.nan, codes) # Optional: Convert -1 back to NaN if needed
             data[col] = codes # Replace original column with codes
             categorical_mappings[col] = {
                 'categories': categories.tolist(),
                 # 'original_values': data[col].unique().tolist() # Storing categories is sufficient
             }
        report['categorical_info'] = categorical_mappings


        # --- 6) Convert ALL non-factorized columns to numeric, coercing errors --- 
        columns_to_coerce = [col for col in data.columns if col not in categorical_mappings]
        
        if not columns_to_coerce:
             # This case happens if ALL columns were factorized (auto mode)
             if not categorical_mappings: # Should not happen if file wasn't empty and parsing worked
                  report['error'] = "No columns left to process after categorical handling."
                  return None, report
             else: # All columns were categorical and factorized
                  data_coerced = pd.DataFrame(index=data.index) # Empty DF for NA check consistency
        else:
             # Important: Apply coercion column by column to handle potential type errors robustly
             data_coerced = pd.DataFrame(index=data.index)
             for col in columns_to_coerce:
                 data_coerced[col] = pd.to_numeric(data[col], errors='coerce')        # --- 7) Handle NA values based on INTENDED numeric columns --- 
        # Check NAs *only* in columns that were originally numeric or looked numeric
        # This ensures string columns in 'preserve' or 'none' modes don't trigger NA detection
        # relevant_cols_for_na = [col for col in intended_numeric_columns if col in data_coerced.columns] # OLD LOGIC

        # if not relevant_cols_for_na or data_coerced.empty: # OLD LOGIC
        #      # No columns considered numeric for NA check OR no columns left after factorizing all
        #      rows_with_na = pd.Series([False] * data.shape[0], index=data.index)
        # else: # OLD LOGIC
        #      # Check for NAs only in the relevant subset of the coerced data
        #      rows_with_na = data_coerced[relevant_cols_for_na].isna().any(axis=1)

        # --- NEW LOGIC: Check all columns in data_coerced for NaNs ---
        if data_coerced.empty:
            rows_with_na = pd.Series([False] * data.shape[0], index=data.index)
        else:
            # Check for NAs across *all* columns resulting from coercion
            rows_with_na = data_coerced.isna().any(axis=1)
        # --- END NEW LOGIC ---

        report['na_handling']['na_detected'] = bool(rows_with_na.any()) # Ensure boolean type

        rows_to_keep = pd.Series([True] * data.shape[0], index=data.index)
        if report['na_handling']['na_detected']:
            if na_policy == 'abort':
                # Updated error message to be more general
                report['error'] = "NA values detected after coercion and na_policy is 'abort'."
                # Find first row/col with NA for better error message
                first_na_row_idx = rows_with_na[rows_with_na].index[0]
                # Check NAs in all coerced columns to find the first error location
                first_na_col = data_coerced.loc[first_na_row_idx].isna().idxmax()
                report['error'] += f" First NA found in column '{first_na_col}' at index {first_na_row_idx}."
                return None, report
            elif na_policy == 'remove':
                rows_to_keep = ~rows_with_na
                report['na_handling']['nb_removed_rows'] = int(rows_with_na.sum()) # Ensure int type
                # Get original indices before potential reset_index
                report['na_handling']['removed_rows_indices'] = data.index[rows_with_na].tolist()
            # else: na_policy is 'ignore' or invalid (already checked), so we keep all rows

        # --- 8) Final conversion to numpy --- 
        final_data_frames = []
        processed_columns_order = [] # Keep track of the order

        # Add factorized columns first, applying row removal
        if categorical_mappings:
            factorized_df = data[list(categorical_mappings.keys())].loc[rows_to_keep]
            final_data_frames.append(factorized_df)
            processed_columns_order.extend(factorized_df.columns)

        # Add coerced columns, applying row removal
        if not data_coerced.empty:
             coerced_df = data_coerced.loc[rows_to_keep]
             final_data_frames.append(coerced_df)
             processed_columns_order.extend(coerced_df.columns)

        if not final_data_frames:
             # Handle case where all rows were removed or no columns processed
             final_shape_cols = len(data.columns) # Number of original columns
             final_shape = (0, final_shape_cols)
             report['final_shape'] = final_shape
             if report['na_handling']['nb_removed_rows'] == report['initial_shape'][0]:
                 report['warnings'].append("All rows removed due to NA values in intended numeric columns.")
             else:
                 report['error'] = "No data left after processing. Check input file and parameters."
                 # Return None if error, empty array if just rows removed
                 if report['error']: return None, report
             # Return empty array with correct number of original columns
             return np.empty(final_shape, dtype=np.float32), report


        # Concatenate factorized and coerced parts
        final_data = pd.concat(final_data_frames, axis=1)

        # Ensure correct column order based on the processed columns list
        # This preserves the relative order from the original file for the columns that were kept
        final_data = final_data[processed_columns_order]

        report['final_shape'] = final_data.shape
        
        # Final check for all-NaN columns which might indicate issues
        all_nan_cols = final_data.columns[final_data.isna().all()].tolist()
        if all_nan_cols:
            report['warnings'].append(f"Columns became all NaN after processing: {all_nan_cols}. This might indicate issues with data type detection or coercion.")
            
        # Convert to numpy float32 array
        try:
            result_array = final_data.astype(np.float32).values
        except Exception as e:
            report['error'] = f"Failed to convert final data to float32 numpy array: {e}"
            return None, report
            
        return result_array, report

    except FileNotFoundError as e:
        report['error'] = str(e)
        return None, report
    except ValueError as e:
        report['error'] = f"ValueError during processing: {e}"
        return None, report
    except Exception as e:
        # Catch any other unexpected error during loading/processing
        import traceback
        report['error'] = f"Unexpected error in load_csv: {e}\n{traceback.format_exc()}"
        return None, report
