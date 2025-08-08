import os
from typing import Optional

# Supported file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'csv', 'db'
}

FILE_TYPE_MAPPING = {
    'pdf': 'PDF Document',
    'docx': 'Word Document', 
    'txt': 'Text File',
    'jpg': 'Image (JPEG)',
    'jpeg': 'Image (JPEG)',
    'png': 'Image (PNG)',
    'bmp': 'Image (BMP)',
    'tiff': 'Image (TIFF)',
    'csv': 'CSV Data',
    'db': 'SQLite Database'
}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename:
        return False
    
    extension = filename.rsplit('.', 1)[-1].lower()
    return extension in ALLOWED_EXTENSIONS

def get_file_extension(filename: str) -> Optional[str]:
    """Get file extension from filename"""
    if not filename or '.' not in filename:
        return None
    
    return filename.rsplit('.', 1)[-1].lower()

def get_file_type(filename: str) -> str:
    """Get human-readable file type"""
    extension = get_file_extension(filename)
    
    if not extension:
        return "Unknown"
    
    return FILE_TYPE_MAPPING.get(extension, f"{extension.upper()} File")

def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
    """Validate file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing potentially dangerous characters"""
    # Keep only alphanumeric, dots, hyphens, and underscores
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    safe_name = ''.join(safe_chars)
    
    # Ensure it's not empty and doesn't start with a dot
    if not safe_name or safe_name.startswith('.'):
        safe_name = 'file_' + safe_name
    
    return safe_name

def ensure_directory(directory_path: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def cleanup_old_files(directory_path: str, max_age_days: int = 7) -> int:
    """Clean up old files in directory"""
    import time
    
    if not os.path.exists(directory_path):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    removed_count = 0
    
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    removed_count += 1
                    
    except Exception as e:
        print(f"Error cleaning up files: {e}")
    
    return removed_count