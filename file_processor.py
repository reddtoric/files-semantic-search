"""
File processing module for semantic file search
Handles file scanning, text extraction, and content processing
"""

import os
import gc
import stat
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Imports that will be available after load_heavy_imports
try:
    from unstructured.partition.auto import partition
except ImportError:
    partition = None

from config import SUPPORTED_EXTENSIONS, SMART_EXCLUDES
from utils import FileCache, ProgressTracker, is_shutdown_requested

logger = logging.getLogger(__name__)


class FileProcessor:
    """Enhanced multi-threaded file processing with performance optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.cache = FileCache() if config.use_cache else None
        self.progress_tracker = ProgressTracker()
        self.file_queue = Queue()
        self.result_queue = Queue()
    
    def _is_hidden_or_system_file(self, file_path: str) -> bool:
        """Check if file has Windows hidden/system attributes"""
        if os.name != 'nt':
            # On Unix-like systems, check for dot files
            return os.path.basename(file_path).startswith('.')
        
        try:
            attrs = os.stat(file_path).st_file_attributes
            return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | 
                               stat.FILE_ATTRIBUTE_SYSTEM | 
                               stat.FILE_ATTRIBUTE_TEMPORARY))
        except (AttributeError, OSError):
            # Fallback: check if filename starts with dot (Unix-style hidden)
            return os.path.basename(file_path).startswith('.')
    
    def _should_skip_directory(self, dir_path: str) -> bool:
        """Check if directory should be skipped"""
        dir_name = os.path.basename(dir_path).lower()
        return (dir_name in self.config.exclude_dirs or 
                self._is_hidden_or_system_file(dir_path))
    
    def _has_supported_extension(self, filename: str) -> bool:
        """Fast extension check using string operations"""
        # Find last dot position
        dot_pos = filename.rfind('.')
        if dot_pos == -1:
            return False
        
        ext = filename[dot_pos:].lower()
        return ext in self.config.include_extensions
    
    def _should_skip_directory_fast(self, dir_name: str) -> bool:
        """Fast directory skip check"""
        dir_lower = dir_name.lower()
        return dir_lower in self.config.exclude_dirs or dir_name.startswith('.')
    
    def _fast_file_scan(self, root_dir: str, debug: bool = False) -> List[str]:
        """Faster file collection using os.scandir instead of os.walk"""
        files_to_process = []
        total_files_seen = 0
        skipped_dirs = set()
        
        def scan_directory(dir_path: str):
            nonlocal total_files_seen
            try:
                with os.scandir(dir_path) as entries:
                    for entry in entries:
                        if is_shutdown_requested():
                            break
                            
                        if entry.is_file(follow_symlinks=False):
                            total_files_seen += 1
                            # Quick extension check first (fastest filter)
                            if self._has_supported_extension(entry.name):
                                # Quick size check for empty files
                                if self.config.skip_empty_files:
                                    try:
                                        if entry.stat().st_size == 0:
                                            continue
                                    except OSError:
                                        continue
                                
                                files_to_process.append(entry.path)
                                
                        elif entry.is_dir(follow_symlinks=False):
                            # Quick directory name check
                            if not self._should_skip_directory_fast(entry.name):
                                scan_directory(entry.path)
                            elif debug:
                                skipped_dirs.add(entry.name)
                                
            except (PermissionError, OSError):
                pass  # Skip inaccessible directories
        
        if debug:
            print(f"DEBUG: Starting optimized file scan of {root_dir}")
        
        scan_directory(root_dir)
        
        if debug:
            print(f"DEBUG: Scanned {total_files_seen} total files, collected {len(files_to_process)} to process")
            if skipped_dirs:
                print(f"DEBUG: Skipped directory types: {sorted(skipped_dirs)}")
        
        return files_to_process
    
    def _should_process_file(self, file_path: str) -> tuple[bool, str]:
        """Determine if file should be processed and return reason if not"""
        
        # Check file extension (already done in fast scan, but double-check)
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.config.include_extensions:
            return False, f"unsupported extension: {file_ext}"
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.config.max_file_size:
                return False, f"file too large: {file_size:,} bytes"
            if file_size == 0 and self.config.skip_empty_files:
                return False, "empty file"
        except OSError as e:
            return False, f"cannot access file: {e}"
        
        # Check if file is hidden/system
        if self._is_hidden_or_system_file(file_path):
            return False, "hidden or system file"
        
        # Check cache if enabled
        if self.cache and not self.cache.is_file_modified(file_path):
            return False, "not modified since last scan"
        
        return True, ""
    
    def _extract_text_from_json(self, file_path: str) -> Optional[str]:
        """Extract text from JSON files by converting to readable format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to searchable text
            def json_to_text(obj, prefix=""):
                """Convert JSON object to searchable text"""
                text_parts = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            text_parts.append(f"{prefix}{key}")
                            text_parts.extend(json_to_text(value, f"{prefix}{key} "))
                        else:
                            text_parts.append(f"{prefix}{key} {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            text_parts.extend(json_to_text(item, prefix))
                        else:
                            text_parts.append(f"{prefix}{item}")
                else:
                    text_parts.append(f"{prefix}{obj}")
                
                return text_parts
            
            content_parts = json_to_text(data)
            content_text = " ".join(content_parts)
            
            if not content_text.strip():
                return None
            
            # Enhanced path and filename context
            filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(filename)[0]
            
            # Get directory path components for context
            dir_path = os.path.dirname(file_path)
            path_components = []
            
            # Extract meaningful directory names (excluding common prefixes)
            try:
                # Get relative path from root directory
                if self.config.root_dir and os.path.isabs(file_path):
                    try:
                        rel_path = os.path.relpath(file_path, self.config.root_dir)
                        if not rel_path.startswith('..'):
                            # Use relative path components
                            path_parts = Path(rel_path).parts[:-1]  # Exclude filename
                            path_components = [part for part in path_parts if part not in SMART_EXCLUDES]
                    except ValueError:
                        pass
                
                # Fallback to absolute path components
                if not path_components:
                    path_parts = Path(dir_path).parts
                    # Take last few meaningful components
                    meaningful_parts = [part for part in path_parts[-3:] if part not in SMART_EXCLUDES]
                    path_components = meaningful_parts
                    
            except Exception:
                # If path processing fails, just use filename
                pass
            
            # Build enhanced context
            context_parts = []
            
            # Add path context
            if path_components:
                path_context = " ".join(path_components)
                context_parts.append(f"Path: {path_context}")
            
            # Add filename context (both with and without extension)
            context_parts.append(f"Filename: {filename}")
            if filename_without_ext != filename:
                context_parts.append(f"Name: {filename_without_ext}")
            
            context_parts.append("Type: json file")
            
            # Add special context for common JSON file types
            if filename.lower() == 'package.json':
                context_parts.append("package configuration nodejs npm yarn dependencies scripts devDependencies")
            elif filename.lower() in ['tsconfig.json', 'jsconfig.json']:
                context_parts.append("typescript javascript configuration compiler options")
            elif filename.lower() in ['composer.json']:
                context_parts.append("php composer dependencies autoload")
            elif filename.lower() in ['manifest.json']:
                context_parts.append("manifest web extension chrome firefox browser")
            elif filename.lower() in ['config.json', 'settings.json']:
                context_parts.append("configuration settings options preferences")
            
            # Combine all context with content
            enhanced_context = " ".join(context_parts)
            combined_text = f"{enhanced_context} Content: {content_text}".strip()
            
            return combined_text
            
        except Exception as e:
            logger.debug(f"Error extracting JSON from {file_path}: {e}")
            return None
    
    def _extract_enhanced_text_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from file with enhanced path/filename context"""
        try:
            # Handle JSON files specially since unstructured doesn't process them properly
            if file_path.lower().endswith('.json'):
                return self._extract_text_from_json(file_path)
            
            # Handle other structured files that unstructured might not process well
            if file_path.lower().endswith(('.csv', '.tsv')):
                return self._extract_text_from_csv(file_path)
            
            # Extract file content using unstructured for other file types
            elements = partition(filename=file_path)
            content_text = " ".join([el.text for el in elements if hasattr(el, "text") and el.text])
            
            if not content_text.strip():
                # Fallback: try reading as plain text if unstructured fails
                return self._extract_text_fallback(file_path)
            
            # Enhanced path and filename context for better search
            filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(filename)[0]
            
            # Get directory path components for context
            dir_path = os.path.dirname(file_path)
            path_components = []
            
            # Extract meaningful directory names (excluding common prefixes)
            try:
                # Get relative path from root directory
                if self.config.root_dir and os.path.isabs(file_path):
                    try:
                        rel_path = os.path.relpath(file_path, self.config.root_dir)
                        if not rel_path.startswith('..'):
                            # Use relative path components
                            path_parts = Path(rel_path).parts[:-1]  # Exclude filename
                            path_components = [part for part in path_parts if part not in SMART_EXCLUDES]
                    except ValueError:
                        pass
                
                # Fallback to absolute path components
                if not path_components:
                    path_parts = Path(dir_path).parts
                    # Take last few meaningful components
                    meaningful_parts = [part for part in path_parts[-3:] if part not in SMART_EXCLUDES]
                    path_components = meaningful_parts
                    
            except Exception:
                # If path processing fails, just use filename
                pass
            
            # Build enhanced context
            context_parts = []
            
            # Add path context
            if path_components:
                path_context = " ".join(path_components)
                context_parts.append(f"Path: {path_context}")
            
            # Add filename context (both with and without extension)
            context_parts.append(f"Filename: {filename}")
            if filename_without_ext != filename:
                context_parts.append(f"Name: {filename_without_ext}")
            
            # Add file extension as a separate searchable term
            file_ext = Path(file_path).suffix.lower()
            if file_ext:
                context_parts.append(f"Type: {file_ext[1:]} file")  # Remove the dot
            
            # Combine all context with content
            enhanced_context = " ".join(context_parts)
            combined_text = f"{enhanced_context} Content: {content_text}".strip()
            
            return combined_text
            
        except Exception as e:
            logger.debug(f"Error extracting text from {file_path}: {e}")
            # Try fallback method
            return self._extract_text_fallback(file_path)
    
    def _extract_text_from_csv(self, file_path: str) -> Optional[str]:
        """Extract text from CSV/TSV files"""
        try:
            import csv
            
            # Detect delimiter
            delimiter = ',' if file_path.lower().endswith('.csv') else '\t'
            
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                # Read first few lines to get structure
                sample = f.read(1024)
                f.seek(0)
                
                # Try to detect if there are headers
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(sample)
                
                reader = csv.reader(f, delimiter=delimiter)
                
                text_parts = []
                row_count = 0
                
                for row in reader:
                    if row_count == 0 and has_header:
                        # Add headers as searchable text
                        text_parts.extend([f"header {header}" for header in row if header.strip()])
                    else:
                        # Add row data
                        text_parts.extend([cell for cell in row if cell.strip()])
                    
                    row_count += 1
                    # Limit to prevent huge files from consuming too much memory
                    if row_count > 1000:
                        break
            
            content_text = " ".join(text_parts)
            
            if not content_text.strip():
                return None
            
            # Add context
            filename = os.path.basename(file_path)
            file_type = "csv" if file_path.lower().endswith('.csv') else "tsv"
            context = f"Filename: {filename} Type: {file_type} file data table spreadsheet"
            
            return f"{context} Content: {content_text}".strip()
            
        except Exception as e:
            logger.debug(f"Error extracting CSV from {file_path}: {e}")
            return None
    
    def _extract_text_fallback(self, file_path: str) -> Optional[str]:
        """Fallback text extraction for when unstructured fails"""
        try:
            # Try reading as plain text with various encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    if content.strip():
                        # Add basic context
                        filename = os.path.basename(file_path)
                        file_ext = Path(file_path).suffix.lower()
                        context = f"Filename: {filename}"
                        if file_ext:
                            context += f" Type: {file_ext[1:]} file"
                        
                        return f"{context} Content: {content}".strip()
                        
                except UnicodeDecodeError:
                    continue
                    
            return None
            
        except Exception as e:
            logger.debug(f"Fallback text extraction failed for {file_path}: {e}")
            return None
    
    def _process_file_worker(self, file_path: str, debug: bool = False) -> Optional[tuple[str, str]]:
        """Worker function to process a single file (thread-safe)"""
        # Check for shutdown request
        if is_shutdown_requested():
            return None
            
        should_process, reason = self._should_process_file(file_path)
        if not should_process:
            self.progress_tracker.increment_skipped()
            file_ext = Path(file_path).suffix.lower()
            if debug and (file_ext in self.config.include_extensions):
                logger.debug(f"Skipped supported file {file_path}: {reason}")
            return None
        
        try:
            text = self._extract_enhanced_text_from_file(file_path)
            if text:
                self.progress_tracker.increment_processed()
                if self.cache:
                    self.cache.update_file_time(file_path)
                
                if debug:
                    processed, _, _ = self.progress_tracker.get_counts()
                    if processed <= 5:  # Show first 5 processed files
                        logger.debug(f"Processed {file_path} ({len(text)} chars)")
                
                return (file_path, text)
            else:
                self.progress_tracker.increment_skipped()
                if debug:
                    logger.debug(f"No text extracted from {file_path}")
                return None
        
        except Exception as e:
            self.progress_tracker.increment_error()
            if debug:
                logger.debug(f"Error processing {file_path}: {e}")
            return None
    
    def _batch_file_processor(self, file_batch: List[str], debug: bool = False) -> List[tuple[str, str]]:
        """Process multiple files in a single thread call"""
        results = []
        
        for file_path in file_batch:
            if is_shutdown_requested():
                break
                
            result = self._process_file_worker(file_path, debug)
            if result:
                results.append(result)
        
        return results
    
    def load_files_from_directory(self, root_dir: str, debug: bool = False) -> Dict[str, str]:
        """Load and process files from directory with optimized performance"""
        file_texts = {}
        
        if debug:
            print(f"DEBUG: Starting scan of directory: {root_dir}")
            print(f"DEBUG: Looking for extensions: {sorted(self.config.include_extensions)}")
            print(f"DEBUG: Max file size: {self.config.max_file_size:,} bytes")
            print(f"DEBUG: Excluded directories: {sorted(self.config.exclude_dirs)}")
            print(f"DEBUG: Using {self.config.threads} worker threads")
            print(f"DEBUG: Fast scan: {self.config.fast_scan}")
        
        # Step 1: Fast file collection
        start_time = time.time()
        if self.config.fast_scan:
            files_to_process = self._fast_file_scan(root_dir, debug)
        else:
            # Fallback to original scanning method
            files_to_process = self._legacy_file_scan(root_dir, debug)
        
        scan_time = time.time() - start_time
        
        if debug:
            print(f"DEBUG: File collection took {scan_time:.2f}s")
        
        if is_shutdown_requested():
            return {}
        
        if not files_to_process:
            return {}
        
        # Step 2: Batch processing for better CPU utilization
        batch_size = max(10, len(files_to_process) // (self.config.threads * self.config.batch_size_factor))
        
        # Sort files to process priority extensions first
        if self.config.priority_extensions:
            priority_files = []
            other_files = []
            
            for file_path in files_to_process:
                ext = Path(file_path).suffix.lower()
                if ext in self.config.priority_extensions:
                    priority_files.append(file_path)
                else:
                    other_files.append(file_path)
            
            files_to_process = priority_files + other_files
            
            if debug:
                print(f"DEBUG: Prioritized {len(priority_files)} files with priority extensions")
        
        # Apply max files limit if specified
        if self.config.max_files_limit:
            files_to_process = files_to_process[:self.config.max_files_limit]
            if debug:
                print(f"DEBUG: Limited to {len(files_to_process)} files")
        
        # Create batches
        file_batches = [files_to_process[i:i + batch_size] 
                       for i in range(0, len(files_to_process), batch_size)]
        
        if debug:
            print(f"DEBUG: Processing in {len(file_batches)} batches of ~{batch_size} files each")
        
        # Step 3: Process batches with thread pool
        processed_files = 0
        try:
            with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
                future_to_batch = {
                    executor.submit(self._batch_file_processor, batch, debug): batch
                    for batch in file_batches
                }
                
                for future in as_completed(future_to_batch):
                    if is_shutdown_requested():
                        # Cancel remaining futures
                        for f in future_to_batch:
                            f.cancel()
                        break
                        
                    try:
                        batch_results = future.result()
                        for file_path, text in batch_results:
                            file_texts[file_path] = text
                            processed_files += 1
                            
                            # Periodic garbage collection for memory management
                            if processed_files % self.config.gc_frequency == 0:
                                gc.collect()
                                
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: Batch processing error: {e}")
        
        except KeyboardInterrupt:
            return {}
        
        if is_shutdown_requested():
            return {}
        
        # Save cache if used
        if self.cache:
            self.cache.save_cache()
        
        processed, errors, skipped = self.progress_tracker.get_counts()
        
        if debug:
            print(f"DEBUG: Processing summary:")
            print(f"  Files collected: {len(files_to_process)}")
            print(f"  Processed: {processed}")
            print(f"  Skipped: {skipped}")
            print(f"  Errors: {errors}")
        
        return file_texts
    
    def _legacy_file_scan(self, root_dir: str, debug: bool = False) -> List[str]:
        """Original file scanning method for fallback"""
        files_to_process = []
        total_files_seen = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Check for shutdown request
                if is_shutdown_requested():
                    break
                    
                # Filter out directories to skip
                original_dirs = dirnames.copy()
                dirnames[:] = [d for d in dirnames if not self._should_skip_directory(os.path.join(dirpath, d))]
                
                if debug and len(original_dirs) != len(dirnames):
                    skipped_dirs = set(original_dirs) - set(dirnames)
                    print(f"DEBUG: Skipping directories in {dirpath}: {skipped_dirs}")
                
                for filename in filenames:
                    # Check for shutdown request
                    if is_shutdown_requested():
                        break
                        
                    total_files_seen += 1
                    file_path = os.path.join(dirpath, filename)
                    files_to_process.append(file_path)
        
        except KeyboardInterrupt:
            pass
            
        return files_to_process