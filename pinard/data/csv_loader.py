# csvloader.py

import csv
import io
from io import StringIO
import numpy as np
import pandas as pd
import re
import gzip
import zipfile
from pathlib import Path


def _can_be_float(value, decimal_sep):
    """Vérifie si une chaîne peut être convertie en float."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False  # Vide n'est pas numérique ici
    try:
        # Remplacer le séparateur décimal potentiel avant la conversion
        # Gérer le cas où le séparateur décimal est '.' (pas besoin de replace)
        if decimal_sep == '.':
            float(value)
        else:
            # Remplacer une seule fois pour éviter les problèmes avec les milliers
            float(value.replace(decimal_sep, '.', 1))
        return True
    except ValueError:
        return False


def _determine_csv_parameters(csv_content: str, sample_lines=20):
    """
    Inspecte les premières lignes d'un contenu CSV (str) pour détecter
    automatiquement le délimiteur, le séparateur décimal et la présence
    d'un header.

    Args:
        csv_content (str): Le contenu du fichier CSV sous forme de chaîne.
        sample_lines (int): Le nombre de lignes à utiliser pour l'analyse.

    Returns:
        dict or None: Un dictionnaire contenant 'delimiter', 'decimal_separator',
                      'has_header', et 'quote_char' si la détection réussit.
                      Retourne None si le contenu est vide ou si la détection échoue.
    """
    lines = []
    # Utiliser io.StringIO pour lire les lignes du contenu comme un fichier
    with io.StringIO(csv_content) as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            # Ignorer les lignes complètement vides
            if line.strip():
                lines.append(line)

    if not lines:
        print("Attention : Le contenu CSV est vide ou ne contient que des lignes vides.")
        return None

    # --- 1. Détection du délimiteur et du caractère de citation ---
    delimiter = None
    quote_char = '"'  # Défaut raisonnable
    try:
        sample_data = "".join(lines)
        dialect = csv.Sniffer().sniff(sample_data)
        delimiter = dialect.delimiter
        quote_char = dialect.quotechar
        # S'assurer que le délimiteur n'est pas alphanumérique
        if delimiter.isalnum():
            print(f"Attention: Le délimiteur détecté par Sniffer ('{delimiter}') est alphanumérique. Tentative manuelle.")
            delimiter = None  # Forcer la détection manuelle ci-dessous
    except csv.Error:
        print("csv.Sniffer a échoué, tentative de détection manuelle du délimiteur...")
        delimiter = None  # Forcer la détection manuelle ci-dessous
    except Exception as e:
        print(f"Erreur inattendue pendant Sniffer : {e}")
        delimiter = None  # Forcer la détection manuelle


    if delimiter is None:
        # Tentative manuelle si Sniffer échoue ou donne un résultat suspect
        possible_delimiters = [';', ',', '\t', '|', ' ']
        best_delim = None
        max_consistent_cols = -1
        most_cols_at_max_consistency = 0

        for delim_candidate in possible_delimiters:
            try:
                # Utiliser io.StringIO pour lire les lignes pour csv.reader
                reader = csv.reader(io.StringIO("".join(lines)), delimiter=delim_candidate, quotechar=quote_char)
                cols_counts = [len(row) for row in reader if row]  # Compte cols des lignes non vides

                if not cols_counts: 
                    continue

                # Le nombre de colonnes le plus fréquent
                most_frequent_cols = max(set(cols_counts), key=cols_counts.count)
                # Nombre de lignes ayant ce nombre de colonnes (cohérence)
                consistency = sum(1 for count in cols_counts if count == most_frequent_cols)

                # Choisir le délimiteur qui maximise la cohérence,
                # puis le nombre de colonnes (ignorer si 1 colonne seulement)
                if most_frequent_cols > 1:
                    if consistency > max_consistent_cols:
                        max_consistent_cols = consistency
                        most_cols_at_max_consistency = most_frequent_cols
                        best_delim = delim_candidate
                    elif consistency == max_consistent_cols:
                        # Si cohérence égale, préférer celui qui donne plus de colonnes
                        if most_frequent_cols > most_cols_at_max_consistency:
                            most_cols_at_max_consistency = most_frequent_cols
                            best_delim = delim_candidate

            except Exception:
                continue  # Ignorer les erreurs de parsing avec ce délimiteur

        if best_delim:
            delimiter = best_delim
            print(f"Délimiteur manuel détecté : '{delimiter}'")
        else:
            print("Erreur : Impossible de déterminer un délimiteur fiable.")
            return None

    # --- 2. Préparation pour la détection du header/décimal ---
    try:
        # Relire avec le bon délimiteur pour obtenir les lignes parsées
        parsed_rows_reader = csv.reader(io.StringIO("".join(lines)), delimiter=delimiter, quotechar=quote_char)
        parsed_rows = [row for row in parsed_rows_reader if any(field.strip() for field in row)]
    except Exception as e:
        print(f"Erreur lors du parsing des lignes avec le délimiteur '{delimiter}': {e}")
        return None


    if not parsed_rows:
        print("Attention : Aucune donnée valide trouvée après parsing initial.")
        return None

    # Utiliser le nombre de colonnes de la première ligne comme référence
    # (ou la majorité si la première est anormale - géré implicitement par la suite)
    num_cols = len(parsed_rows[0])
    if num_cols == 0:
        print("Attention : La première ligne semble vide ou invalide après parsing.")
        return None  # ou essayer de trouver une ligne valide?

    # --- 3. Détection Header / Séparateur Décimal ---
    best_decimal_sep = '.'  # Défaut
    best_has_header = False
    max_numeric_score = -1.0  # Utiliser float pour comparaison

    for decimal_sep in ['.', ',']:
        for has_header_option in [False, True]:
            # Déterminer les lignes de données à tester
            first_data_row_index = 1 if has_header_option else 0

            # Vérifier qu'il y a assez de lignes pour l'option choisie
            if len(parsed_rows) <= first_data_row_index:
                # Pas assez de lignes pour avoir des données avec cette option header
                if has_header_option and len(parsed_rows) == 1:
                    # Cas spécial: 1 ligne, on suppose que c'est un header
                    # Le score sera 0 car pas de données à évaluer
                    numeric_cells = 0
                    total_cells = 0
                    current_score = 0.0
                else:  # Pas de données ou 0 lignes
                    continue  # Impossible d'évaluer
            else:
                data_rows = parsed_rows[first_data_row_index:]
                numeric_cells = 0
                total_cells = 0

                for row in data_rows:
                    # Considérer uniquement les lignes avec un nombre ~ cohérent de colonnes
                    # (Tolérance simple ici, pourrait être affinée)
                    if abs(len(row) - num_cols) <= 1:  # Permet une petite variation
                        effective_num_cols = len(row)  # Utiliser la longueur réelle de la ligne
                        for i in range(effective_num_cols):
                            total_cells += 1
                            if _can_be_float(row[i], decimal_sep):
                                numeric_cells += 1

                if total_cells == 0:
                    current_score = 0.0
                else:
                    current_score = numeric_cells / total_cells

            # Pénalité si on suppose un header mais qu'il est très numérique
            # et que les données sont aussi numériques.
            if has_header_option and len(parsed_rows) > 0:
                header_row = parsed_rows[0]
                # S'assurer que la ligne header a aussi le bon nombre de colonnes
                if len(header_row) == num_cols:
                    header_numeric_cells = 0
                    header_total_cells = 0
                    for cell in header_row:
                        header_total_cells += 1
                        if _can_be_float(cell, decimal_sep):
                            header_numeric_cells += 1

                    header_score = 0.0
                    if header_total_cells > 0:
                        header_score = header_numeric_cells / header_total_cells

                    # Appliquer la pénalité si le header est >= aussi numérique que les données
                    # sauf si les données elles-mêmes ne sont pas très numériques
                    if current_score > 0.5 and header_score >= current_score:
                        # Réduire le score de moitié si header semble être des données
                        current_score *= 0.5

            # Mettre à jour si ce score est le meilleur trouvé
            # Ajouter une petite marge pour éviter de changer pour des scores très similaires
            if current_score > max_numeric_score + 1e-6:
                max_numeric_score = current_score
                best_decimal_sep = decimal_sep
                best_has_header = has_header_option
            # Gérer égalité: préférer '.' et pas de header si scores égaux
            elif abs(current_score - max_numeric_score) < 1e-6:
                if best_decimal_sep == ',' and decimal_sep == '.':
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option
                elif best_has_header is True and has_header_option is False:
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option

    # Si le score numérique est très bas, la détection est peu fiable
    if max_numeric_score < 0.5:  # Seuil de 50% de cellules numériques dans les données
        print(f"Attention : Faible score de numéricité ({max_numeric_score:.2f}). "
              "Le fichier ne semble pas contenir majoritairement de nombres après le header potentiel.")
        # Retourner le meilleur résultat trouvé, mais l'utilisateur doit être prudent

    return {
        'delimiter': delimiter,
        'decimal_separator': best_decimal_sep,
        'has_header': best_has_header,
        'quote_char': quote_char,
        'confidence': max_numeric_score  # Ajouter le score pour information
    }

# =============================================================================
# Fonction principale de chargement CSV (modifiée)
# =============================================================================

def load_csv(path, na_policy='auto', type='x'):  # Garde 'type' pour compatibilité mais non utilisé ici pour la détection
    """
    Charge un fichier CSV, détecte ses propriétés, le nettoie et le retourne
    en tant que tableau NumPy float32.

    Args:
        path (str ou Path): Chemin vers le fichier CSV (.csv, .gz, .zip).
        na_policy (str): Politique de gestion des NA ('abort', 'auto').
                         'auto' est traité comme 'abort'.
        type (str): Type de données ('x' ou 'y') - conservé pour compatibilité
                      ascendante mais non utilisé dans la détection actuelle.

    Returns:
        tuple (np.ndarray | None, dict):
            - Le tableau NumPy contenant les données nettoyées, ou None si erreur.
            - Un dictionnaire 'report' détaillant le processus de chargement.
    """
    if na_policy == 'auto':
        na_policy = 'abort'
    if na_policy != 'abort':
        # Pour l'instant, seule 'abort' est activement implémentée ici
        raise ValueError("Invalid NA policy - only 'abort' is supported for now.")
        # Si d'autres politiques sont réactivées, ajuster la logique NA ci-dessous.

    report = {
        'file_path': str(path),
        'detection_params': None,  # Pour stocker les params détectés
        'initial_shape': None,
        'final_shape': None,
        'na_handling': {
            'strategy': na_policy,
            'na_detected': False,
            'nb_removed_rows': 0, # Sera > 0 seulement si la politique changeait
            'removed_rows_indices': []  # Sera rempli seulement si la politique changeait
        },
        'error': None
    }

    try:
        # --- 1. Lire le contenu du fichier ---
        content = None
        file_path = Path(path)  # Utiliser pathlib
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas : {path}")

        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:  # rt = read text
                content = f.read()
        elif file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                # Suppose qu'il n'y a qu'un seul fichier CSV pertinent dans le zip
                csv_files_in_zip = [name for name in z.namelist() if name.lower().endswith('.csv')]
                if not csv_files_in_zip:
                    raise ValueError(f"Aucun fichier .csv trouvé dans l'archive ZIP : {path}")
                if len(csv_files_in_zip) > 1:
                    print(f"Attention : Plusieurs fichiers .csv trouvés dans {path}. Utilisation de : {csv_files_in_zip[0]}")
                content = z.read(csv_files_in_zip[0]).decode('utf-8')
        else:  # Suppose plain text (csv, txt, etc.)
            # Essayer utf-8, puis latin-1 comme fallback courant
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"Attention : Échec de la lecture de {path} en UTF-8. Tentative avec Latin-1.")
                try:
                    with open(file_path, 'r', newline='', encoding='latin-1') as f:
                        content = f.read()
                except Exception as enc_err:
                    raise ValueError(f"Impossible de lire le fichier {path} avec UTF-8 ou Latin-1: {enc_err}")

        if content is None or not content.strip():
            raise ValueError("Le fichier est vide ou n'a pas pu être lu.")

        # --- 2. Détecter les paramètres CSV ---
        # Utiliser un nombre de lignes plus important pour la détection si possible
        num_lines_in_file = content.count('\n') + 1
        sample_size = min(50, num_lines_in_file)  # Augmenté à 50 lignes max
        params = _determine_csv_parameters(content, sample_lines=sample_size)

        if params is None:
            raise ValueError("Impossible de déterminer les paramètres CSV (délimiteur, décimal, header).")

        report['detection_params'] = params
        report['delimiter'] = params['delimiter']  # Garder pour compatibilité rapport simple
        report['numeric_delimiter'] = params['decimal_separator']  # Garder pour compatibilité
        report['header_line'] = 0 if params['has_header'] else None  # Garder pour compatibilité

        # --- 3. Charger avec Pandas en utilisant les paramètres détectés ---
        # Utiliser StringIO pour lire le contenu comme un fichier
        data = pd.read_csv(
            StringIO(content),
            header=report['header_line'],
            sep=params['delimiter'],
            decimal=params['decimal_separator'],
            quotechar=params['quote_char'],
            na_filter=False,        # Important pour gérer les NA manuellement après
            engine='python',        # Souvent plus robuste pour des formats variés
            skip_blank_lines=True   # Ignorer les lignes complètement vides
        )
        report['initial_shape'] = data.shape
        # print("Initial shape:", data.shape)

        # --- 4. Nettoyage et validation ---

        # Remplacer les chaînes potentiellement vides ou espaces blancs par NaN avant conversion
        data = data.replace(r'^\s*$', np.nan, regex=True)

        # Convertir tout en numérique, mettre NaN si échec ('coerce')
        # Appliquer colonne par colonne pour potentiellement mieux gérer les erreurs
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # print("Shape after to_numeric:", data.shape)
        # print("NA count after coerce:", data.isna().sum().sum())

        # Supprimer colonnes ou lignes entièrement vides/NA (créées par ex par des virgules finales)
        data = data.dropna(axis=1, how='all')
        data = data.dropna(axis=0, how='all')
        # print("Shape after drop all NA:", data.shape)

        # --- 5. Gérer les NAs restants selon la politique ---
        na_mask = data.isna()
        if na_mask.sum().sum() > 0:
            report['na_handling']['na_detected'] = True
            if na_policy == 'abort':
                rows_with_na_indices = data[na_mask.any(axis=1)].index.tolist()
                report['na_handling']['removed_rows_indices'] = rows_with_na_indices  # Techniquement pas supprimées, mais détectées
                report['na_handling']['nb_removed_rows'] = len(rows_with_na_indices)
                raise ValueError(f"NA values found in data and na_policy is 'abort'. Indices (relatifs au DF après header): {rows_with_na_indices}")
            # elif na_policy == 'remove_rows': # Exemple si on ajoutait cette politique
            #     rows_with_na_indices = data[na_mask.any(axis=1)].index.tolist()
            #     report['na_handling']['removed_rows_indices'] = rows_with_na_indices
            #     report['na_handling']['nb_removed_rows'] = len(rows_with_na_indices)
            #     data = data.dropna(axis=0, how='any')
            #     print(f"Removed {report['na_handling']['nb_removed_rows']} rows containing NA values.")
            # else:
                # Implémenter d'autres politiques ici (fillna, etc.) si nécessaire
            #    pass

        # Si on arrive ici, soit il n'y avait pas de NA, soit la politique permettait de continuer

        report['final_shape'] = data.shape

        # --- 6. Convertir en NumPy float32 ---
        if data.empty:
            print(f"Attention : Le DataFrame est vide après nettoyage pour le fichier {path}.")
            # Retourner un tableau vide de la bonne dimensionnalité ? (0 lignes, N colonnes?)
            # Ou juste None? Retourner None semble plus sûr.
            raise ValueError("Les données sont vides après nettoyage.")

        try:
            # S'assurer que tout est bien flottant avant la conversion finale
            data_np = data.astype(np.float32).values
        except Exception as final_conv_err:
            # Cette erreur peut survenir si des objets non numériques persistent
            print(f"Erreur lors de la conversion finale en NumPy float32: {final_conv_err}")
            print("Types de données Pandas avant conversion:", data.dtypes)
            raise ValueError(f"Conversion finale en np.float32 échouée. Vérifier les données. Erreur: {final_conv_err}")

        # print(f"Parsing csv report for {path}", report)
        return data_np, report

    except Exception as e:
        import traceback  # Pour un débogage plus facile
        print(f"Erreur détaillée dans load_csv pour {path}:")
        print(traceback.format_exc())  # Imprime la trace complète de l'erreur
        report['error'] = str(e)
        # print(f"Report d'erreur pour {path}:", report)
        return None, report