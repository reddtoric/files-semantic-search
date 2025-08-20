"""
Vector database management for semantic file search
Handles embedding generation, database operations, and search functionality
"""

import os
import tempfile
import shutil
import logging
from typing import Dict, List, Tuple
import time

# Imports that will be available after load_heavy_imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    chromadb = None
    SentenceTransformer = None
    torch = None

from utils import ProgressTracker, is_shutdown_requested

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Enhanced vector database management with improved GPU detection and progress tracking"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.client = None
        self.collection = None
        self.device = None
        self.db_progress_tracker = ProgressTracker()  # For database building progress
    
    def get_progress_tracker(self):
        """Get progress tracker for database building"""
        return self.db_progress_tracker
    
    def _get_detailed_gpu_info(self):
        """Get detailed GPU information for debugging"""
        if not torch:
            return {'cuda_available': False}
            
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'gpu_count': 0,
            'gpus': []
        }
        
        if torch.cuda.is_available():
            gpu_info['gpu_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                gpu_info['gpus'].append({
                    'id': i,
                    'name': gpu_name,
                    'memory_gb': gpu_memory
                })
        
        return gpu_info
    
    def _get_best_device(self):
        """Automatically select the best available device with detailed diagnostics"""
        gpu_info = self._get_detailed_gpu_info()
        
        print(f"🔍 HARDWARE DETECTION")
        print("=" * 30)
        print(f"PyTorch version: {gpu_info['torch_version']}")
        print(f"CUDA available: {gpu_info['cuda_available']}")
        
        if gpu_info['cuda_available']:
            print(f"CUDA version: {gpu_info['cuda_version']}")
            print(f"GPU count: {gpu_info['gpu_count']}")
            
            for gpu in gpu_info['gpus']:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
            
            if self.config.use_gpu:
                # Test GPU functionality
                try:
                    # Test basic CUDA operations
                    test_tensor = torch.randn(100, 100).cuda()
                    test_result = torch.mm(test_tensor, test_tensor)
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    
                    print("✅ GPU test successful - using GPU acceleration")
                    print("=" * 30)
                    return 'cuda'
                    
                except Exception as e:
                    print(f"❌ GPU test failed: {e}")
                    print("🔄 Falling back to CPU")
                    print("=" * 30)
                    return 'cpu'
            else:
                print("⚠️  GPU acceleration disabled by user")
                print("=" * 30)
                return 'cpu'
        else:
            print("❌ No CUDA-capable GPU found")
            
            # Check if this is a PyTorch installation issue
            if 'cpu' in torch.__version__:
                print("ℹ️  This appears to be a CPU-only PyTorch installation")
                print("ℹ️  For GPU support, install with:")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
            print("🔄 Using CPU")
            print("=" * 30)
            return 'cpu'
    
    def initialize(self):
        """Initialize the embedding model and database with GPU support"""
        self.device = self._get_best_device()
        
        print(f"🤖 Loading AI model: {self.config.model_name}")
        print(f"   Device: {self.device.upper()}")
        
        # Load model with device specification
        model_start_time = time.time()
        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.device)
            
            # Test model loading
            test_text = "This is a test sentence."
            with torch.no_grad():
                test_embedding = self.model.encode(test_text, convert_to_numpy=True)
            
            model_load_time = time.time() - model_start_time
            print(f"✅ Model loaded successfully ({test_embedding.shape[0]} dimensions, {model_load_time:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            if self.device == 'cuda':
                print("🔄 Retrying with CPU...")
                self.device = 'cpu'
                self.model = SentenceTransformer(self.config.model_name, device=self.device)
                print("✅ Model loaded on CPU")
        
        # Optimize for inference
        if self.device == 'cuda':
            self.model.eval()
            # Enable mixed precision for faster inference
            torch.backends.cudnn.benchmark = True
            
            # Show GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_cached = torch.cuda.memory_reserved() / (1024**2)  # MB
                print(f"📊 GPU memory: {memory_allocated:.1f} MB allocated, {memory_cached:.1f} MB cached")
        
        # Initialize ChromaDB
        db_path = self.config.db_dir if self.config.persist_db else tempfile.mkdtemp()
        self.client = chromadb.PersistentClient(path=db_path)
        print()  # Empty line after initialization
    
    def build_database(self, file_texts: Dict[str, str], force_rebuild: bool = False) -> str:
        """Build vector database from file texts with GPU-optimized batching and progress tracking"""
        if not file_texts:
            raise ValueError("No files to process")
        
        # Check for shutdown request
        if is_shutdown_requested():
            return ""
        
        # Create or get collection
        collection_name = "documents"
        
        if not force_rebuild:
            try:
                self.collection = self.client.get_collection(collection_name)
                # Check if collection has any documents
                if self.collection.count() > 0:
                    return collection_name  # Use existing database
            except Exception:
                pass  # Collection doesn't exist, will create new one
        
        # Create new collection (this will replace existing if it exists)
        try:
            if not force_rebuild:
                self.collection = self.client.create_collection(collection_name)
            else:
                # Force recreation
                try:
                    self.client.delete_collection(collection_name)
                except:
                    pass
                self.collection = self.client.create_collection(collection_name)
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection(collection_name)
        
        # Optimized batch size for GPU vs CPU
        if self.device == 'cuda':
            batch_size = self.config.gpu_batch_size
        else:
            batch_size = 100
        
        file_items = list(file_texts.items())
        total_batches = (len(file_items) + batch_size - 1) // batch_size
        
        print(f"🔧 Building search index: {len(file_items)} documents, {total_batches} batches")
        print(f"   Batch size: {batch_size} (optimized for {self.device.upper()})")
        
        # Reset progress tracker for database building
        self.db_progress_tracker = ProgressTracker()
        
        for batch_idx in range(0, len(file_items), batch_size):
            # Check for shutdown request
            if is_shutdown_requested():
                return ""
                
            batch = file_items[batch_idx:batch_idx + batch_size]
            
            # Prepare batch data
            ids = [str(batch_idx + i) for i in range(len(batch))]
            texts = [text for _, text in batch]
            paths = [path for path, _ in batch]
            
            # Generate embeddings with GPU optimization
            try:
                with torch.no_grad():  # Disable gradient computation for inference
                    embeddings = self.model.encode(
                        texts, 
                        show_progress_bar=False,
                        batch_size=batch_size,
                        convert_to_numpy=True,  # Convert to numpy for ChromaDB
                        normalize_embeddings=True  # Normalize for better similarity
                    ).tolist()
            except Exception as e:
                print(f"❌ Error generating embeddings: {e}")
                if self.device == 'cuda':
                    print("🔄 Retrying batch on CPU...")
                    # Temporarily switch to CPU for this batch
                    cpu_model = SentenceTransformer(self.config.model_name, device='cpu')
                    embeddings = cpu_model.encode(
                        texts, 
                        show_progress_bar=False,
                        batch_size=50,  # Smaller batch for CPU
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).tolist()
                    del cpu_model
                else:
                    raise
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=[{"path": path} for path in paths],
                documents=texts
            )
            
            # Update progress tracker
            for _ in range(len(batch)):
                self.db_progress_tracker.increment_processed()
            
            # Clear GPU cache periodically
            if self.device == 'cuda' and batch_idx % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        return collection_name
    
    def search(self, query: str, top_k: int, min_score: float = 0.1, debug: bool = False) -> List[Tuple[str, float]]:
        """Search with GPU-accelerated query embedding"""
        if not self.collection:
            raise ValueError("Database not initialized")
        
        # Check for shutdown request
        if is_shutdown_requested():
            return []
        
        # Get more results than needed to filter by score
        search_k = min(top_k * 3, self.collection.count())  # Get 3x more to filter
        
        # Generate query embedding on GPU
        try:
            with torch.no_grad():
                query_embedding = self.model.encode(
                    query, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            if self.device == 'cuda':
                print("🔄 Retrying query on CPU...")
                cpu_model = SentenceTransformer(self.config.model_name, device='cpu')
                query_embedding = cpu_model.encode(
                    query, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                del cpu_model
            else:
                raise
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k
        )
        
        if not results["metadatas"] or not results["metadatas"][0]:
            return []
        
        # Combine paths with scores and filter by minimum score
        file_results = []
        all_results = []  # For debugging
        
        for path, distance in zip(
            [m["path"] for m in results["metadatas"][0]], 
            results["distances"][0]
        ):
            # Convert distance to similarity score (closer to 1 is better)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            similarity = 1.0 - distance
            all_results.append((path, similarity))
            
            if similarity >= min_score:
                file_results.append((path, similarity))
        
        # Debug output if requested
        if debug:
            print(f"\n🔍 DEBUG: Search Analysis")
            print(f"Query: '{query}'")
            print(f"Total results found: {len(all_results)}")
            print(f"Results above threshold ({min_score:.3f}): {len(file_results)}")
            print(f"\nTop 10 results (all scores):")
            for i, (path, score) in enumerate(sorted(all_results, key=lambda x: x[1], reverse=True)[:10]):
                status = "✅ PASS" if score >= min_score else "❌ FILTERED"
                print(f"  {i+1:2d}. [{score:.3f}] {status} {os.path.basename(path)}")
        
        # Sort by similarity score (highest first) and limit to top_k
        file_results.sort(key=lambda x: x[1], reverse=True)
        return file_results[:top_k]
    
    def cleanup(self):
        """Clean up resources"""
        if torch and self.device == 'cuda':
            torch.cuda.empty_cache()
            
        if not self.config.persist_db and self.client:
            try:
                # Clean up temporary directory
                db_path = self.client._settings.persist_directory
                if db_path and os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    logger.info("Cleaned up temporary database")
            except Exception as e:
                logger.warning(f"Error cleaning up database: {e}")
