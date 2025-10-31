
import os

def categorize_file(file_path):
    """
    Categorize a file based on its path and extension.
    Returns a tuple of (category, necessary).
    """
    # Define categories based on file extensions and paths
    if file_path.endswith('.md') or file_path.endswith('.txt') or file_path.endswith('.json') or file_path.endswith('.yaml') or file_path.endswith('.bib'):
        if 'docs' in file_path or 'README' in file_path or 'research_log' in file_path or 'citation' in file_path or 'manifest' in file_path:
            return 'Documentation', True
        elif 'validation' in file_path:
            return 'Documentation', True
        else:
            return 'Documentation', False
    elif file_path.endswith('.py') or file_path.endswith('.sh') or file_path.endswith('.ipynb'):
        if 'scripts' in file_path or 'flab_framework' in file_path or 'ml_algorithms' in file_path or 'semantic_mesh' in file_path or 'validation_systems' in file_path:
            return 'Code', True
        else:
            return 'Code', False
    elif file_path.endswith('.csv') or file_path.endswith('.json') or file_path.endswith('.pkl') or file_path.endswith('.joblib'):
        if 'data' in file_path or 'features' in file_path or 'targets' in file_path or 'predictions' in file_path or 'results' in file_path or 'submissions' in file_path or 'models' in file_path:
            return 'Data', True
        else:
            return 'Data', False
    elif file_path.endswith('.tex'):
        return 'Research', True
    elif file_path.endswith('.pyc'):
        return 'Cache', False
    else:
        return 'Other', False

def main():
    """
    Main function to read audit results, categorize files, and write to categorized files.
    """
    with open('/a0/bitcore/workspace/audit_results.txt', 'r') as audit_file, open('/a0/bitcore/workspace/categorized_files.txt', 'w') as categorized_file:
        for line in audit_file:
            line = line.strip()
            if line and not line.startswith('Citation Directory Files:') and not line.startswith('Data Directory Files:') and not line.startswith('Docs Directory Files:') and not line.startswith('Features Directory Files:') and not line.startswith('FLAb Framework Directory Files:') and not line.startswith('ML Algorithms Directory Files:') and not line.startswith('Research Directory Files:') and not line.startswith('Scripts Directory Files:') and not line.startswith('Models Directory Files:') and not line.startswith('Semantic Mesh Directory Files:') and not line.startswith('Submissions Directory Files:') and not line.startswith('Validation Systems Directory Files:'):
                category, necessary = categorize_file(line)
                categorized_file.write(f'{line} | {category} | {necessary}
')

if __name__ == '__main__':
    main()
