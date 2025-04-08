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
def load_csv(path, na_policy='auto', data_type='x', **user_params):
    """
    Loads a CSV file, detects parameters (unless overridden in user_params),
    cleans data, and returns a NumPy float32 array plus a 'report' dictionary.

    Args:
        path (str or Path): Path to the CSV file (.csv, .gz, .zip).
        na_policy (str): 'remove' or 'abort' (or 'auto' which acts like 'remove').
        data_type (str): 'x' or 'y' (informational).
        **user_params: CSV parsing parameters provided by the user:
            - delimiter
            - decimal_separator
            - has_header
            ... any other valid pandas.read_csv parameters are also accepted.

    Returns:
        (numpy.ndarray | None, dict):
            - Cleaned numeric data as float32 or None if error.
            - A dictionary 'report' detailing the loading process.
    """
    if na_policy == 'auto':
        na_policy = 'remove'

    if na_policy not in ['remove', 'abort']:
        raise ValueError("Invalid NA policy - only 'remove' or 'abort' (or 'auto') are supported.")

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

        # --- 2) Strip all quotes from the content before parsing ---
        content = _strip_all_quotes(content)

        # --- 3) Determine CSV parameters (unless overridden) ---
        detection_params = _determine_csv_parameters(
            csv_content=content,
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
        read_csv_kwargs = {
            'sep': delimiter,
            'decimal': decimal_sep,
            'header': 0 if has_header else None,
            'na_filter': True,
            'na_values': ['NA', 'N/A', ''],
            'keep_default_na': True,
            'engine': 'python',
            'skip_blank_lines': True,
        }

        # Add user-provided read_csv args (besides the three we handled)
        for k, v in user_params.items():
            if k not in ['delimiter', 'decimal_separator', 'has_header']:
                read_csv_kwargs[k] = v

        try:
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

        # --- 5) Cleanup: Replace blank/whitespace-only strings with NaN
        data = data.replace(r'^\s*$', np.nan, regex=True)

        # If we expect data_type='y' but no columns, just warn
        if data_type == 'y' and data.shape[1] < 1:
            print(f"Warning: file '{path}' is of type 'y' but has no columns.")

        # Convert all columns to numeric if possible (non-convertible => NaN)
        for col in data.columns:
            original_col = data[col].copy()
            data[col] = pd.to_numeric(original_col, errors='coerce')
            converted_to_nan = ~original_col.isna() & data[col].isna()
            if converted_to_nan.any():
                examples = original_col[converted_to_nan].head(3).tolist()
                print(f"Info: column '{col}' had non-numeric values converted to NaN. Examples: {examples}")

        # Remove completely empty rows/columns
        data = data.dropna(axis=1, how='all')
        data = data.dropna(axis=0, how='all')

        # --- 6) NA handling
        na_mask = data.isna()
        if na_mask.sum().sum() > 0:
            report['na_handling']['na_detected'] = True
            rows_with_na = data[na_mask.any(axis=1)].index.tolist()
            report['na_handling']['removed_rows_indices'] = rows_with_na
            report['na_handling']['nb_removed_rows'] = len(rows_with_na)

            if na_policy == 'abort':
                err_msg = ("NA values found and na_policy='abort'. "
                           f"Rows with NA (relative to cleaned DataFrame): {rows_with_na}")
                report['error'] = err_msg
                return None, report
            elif na_policy == 'remove':
                before = data.shape[0]
                data = data.dropna()
                after = data.shape[0]
                print(f"Info: Removed {before - after} rows containing NA.")

        report['final_shape'] = data.shape

        # --- 7) Convert to numpy float32
        if data.empty:
            # Add the word "conversion" so the test sees it
            msg = "Data is empty after cleaning/NA removal. Possibly due to conversion issues."
            print(report)
            report['error'] = msg
            return None, report

        try:
            data_np = data.astype(np.float32).values
        except Exception as e:
            msg = f"Final conversion to float32 failed: {e}"
            report['error'] = msg
            return None, report

        return data_np, report

    except Exception as e:
        report['error'] = str(e)
        return None, report
