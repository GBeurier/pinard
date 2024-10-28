from io import StringIO
import numpy as np
import pandas as pd
import re


def analyze_csv_file(csv_content: str, type='x'):
    try:
        first_lines = read_first_lines(csv_content, 5)
        delimiter = detect_delimiter(first_lines, type)
        numeric_delimiter = detect_numeric_delimiter(first_lines[1:], type)
        header = detect_column_header(first_lines, delimiter, numeric_delimiter, type)
        return delimiter, numeric_delimiter, header
    except Exception as e:
        raise ValueError(f"Error analyzing CSV file: {str(e)}")


def read_first_lines(csv_content: str, num_lines):
    lines = csv_content.splitlines()[:num_lines]
    if len(lines) < num_lines:
        raise ValueError("File has fewer lines than expected.")
    return lines


def detect_delimiter(lines, type='x'):
    delimiter_candidates = [';', '\t', ',']
    delimiter_counts = {delimiter: 0 for delimiter in delimiter_candidates}

    for line in lines:
        for delimiter in delimiter_candidates:
            delimiter_counts[delimiter] += len(re.findall(re.escape(delimiter), line))

    detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    if delimiter_counts[detected_delimiter] == 0 and type == 'x':
        raise ValueError("Unable to detect a valid delimiter.")
    return detected_delimiter


def detect_numeric_delimiter(lines, type='x'):
    numeric_delimiters = ['.', ',']
    numeric_delimiter_counts = {numeric_delimiter: 0 for numeric_delimiter in numeric_delimiters}

    for line in lines:
        for numeric_delimiter in numeric_delimiters:
            numeric_delimiter_counts[numeric_delimiter] += len(re.findall(re.escape(numeric_delimiter), line))

    detected_numeric_delimiter = max(numeric_delimiter_counts, key=numeric_delimiter_counts.get)
    if numeric_delimiter_counts[detected_numeric_delimiter] == 0 and type == 'x':
        raise ValueError("Unable to detect a valid numeric delimiter.")
    return detected_numeric_delimiter


def detect_column_header(lines, delimiter, numeric_delimiter, type='x'):
    if type == 'y':
        column_name_pattern = fr'^[^\d{re.escape(numeric_delimiter)}]+$'
        column_header_present = any(re.search(column_name_pattern, cell.strip()) for cell in lines[0].split(delimiter))
        # print("column_header_present", column_header_present)
        return 0 if column_header_present else None
    
    column_name_pattern = r'^[\w\d_]+$'
    column_header_present = all(re.match(column_name_pattern, cell.strip()) for cell in lines[0].split(delimiter))
    # print("Before test >, column_header_present", column_header_present)
    if not lines[1].startswith('0') and not lines[1].startswith('1'):
        column_header_present = True
        
    if not isinstance(lines[0].split(delimiter)[0], float):
        column_header_present = True
    
    
    return 0 if column_header_present else None


def load_csv(path, na_policy='auto', type='x'):
    if na_policy == 'auto':
        na_policy = 'abort'
    if not na_policy == 'abort':
        raise ValueError("Invalid NA policy - only 'abort' is supported for now.")
    
    # Step 1: Initialize an empty dictionary for the report.
    report = {
        'initial_shape': None,
        'delimiter': None,
        'numeric_delimiter': None,
        'header_line': None,
        'final_shape': None,
        'na_handling': {
            'strategy': na_policy,
            'nb_removed_rows': None,
            'removed_rows': None
        }
    }
    
    try:
        if path.endswith('.gz'):
            import gzip
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                content = f.read()
                f.close()
        elif path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(path, 'r') as z:
                content = z.read(z.namelist()[0]).decode('utf-8')
                z.close()
        else:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                content = f.read()
                f.close()

        delimiter, nb_delimiter, header = analyze_csv_file(content, type)
        report['delimiter'] = delimiter
        report['numeric_delimiter'] = nb_delimiter
        report['header_line'] = header

        data = pd.read_csv(StringIO(content), header=header, na_filter=False, sep=delimiter, engine='python', skip_blank_lines=False, decimal=nb_delimiter)
        report['initial_shape'] = data.shape
        # print(data.shape)

        # Remove empty lines
        # print(data.shape)
        data = data.replace(r"^\s*$", "", regex=True)
        # print(data.shape)
        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Drop empty columns and rows
        # print(data.shape)
        data = data.dropna(how='all', axis=1)
        # print(data.shape)
        data = data.dropna(how='all', axis=0)
        # print(data.shape)
        # Detect rows with any NaN values
        if data.isna().sum().sum() > 0:
            rows_with_na = data.isna().any(axis=1)
            indexes_with_na = data[rows_with_na].index.tolist()
            report['na_handling']['nb_removed_rows'] = len(indexes_with_na)
            report['na_handling']['removed_rows'] = indexes_with_na
            raise ValueError("NA values found")
        
        # if len(indexes_with_na) > 0:
            # print(indexes_with_na)
            # report['error'] = "NA values found"
            # print(rows_with_na)

        # # Handle NaN based on policy
        # if na_policy == 'abort':
        #     if len(rows_with_na) > 0 or len(indexes_with_na) > 0:
        #         raise ValueError("NA values found")
        # elif na_policy == 'remove':
        #     data = data.drop(indexes_with_na)
        # elif na_policy == 'ignore':
        #     pass
        # elif na_policy == 'replace':
        #     data = data.fillna(0)
        # elif na_policy == 'auto':
        #     threshold = 0.05
        #     rows_to_drop = data.index[data.isna().mean(axis=1) > threshold].tolist()
        #     data = data.drop(rows_to_drop)
        #     indexes_with_na = rows_to_drop
        #     data = data.fillna(0)

        report['final_shape'] = data.shape
        
        # Convert data to numpy array
        data = data.astype(np.float32).values

        # Step 3: Return the processed data and the report.
        # print("Parsing csv report for", path, report)
        return data, report

    except Exception as e:
        report['error'] = str(e)
        print(report)
        return None, report
