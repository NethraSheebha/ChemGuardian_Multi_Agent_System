#!/usr/bin/env python3
"""
Chemical Knowledge Ingestion Pipeline
Processes MSDS PDFs and stores embeddings in Qdrant for chemical leak detection
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
import hashlib
import time
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from utils.pdf_processor import PDFProcessor
from utils.embedder import ChemicalEmbedder


# Setup logging
def setup_logging():
    """Configure logging to both file and console"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ingestion.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)


class MSDSIngestionPipeline:
    """Main pipeline for ingesting MSDS documents into Qdrant"""
    
    def __init__(
        self,
        data_dir: str = "data/msds",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "chemical_docs"
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            data_dir: Directory containing MSDS PDF files
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Target Qdrant collection
        """
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        
        # Verify data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist. Creating it...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.pdf_processor = PDFProcessor(chunk_size=400)
        self.embedder = ChemicalEmbedder(model_name="BAAI/bge-base-en")
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Verify collection exists
        self._verify_collection()
        
        logger.info("Pipeline initialized successfully")
    
    def _verify_collection(self):
        """Verify that the target collection exists"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' found")
            logger.info(f"Points: {collection_info.points_count}")
            logger.info(f"Vector dimension: {collection_info.config.params.vectors.size}")
            
            # Verify dimension matches
            expected_dim = self.embedder.get_dimension()
            actual_dim = collection_info.config.params.vectors.size
            
            if expected_dim != actual_dim:
                raise ValueError(
                    f"Dimension mismatch! Embedder: {expected_dim}, Collection: {actual_dim}"
                )
            
        except Exception as e:
            logger.error(f"Collection verification failed: {e}")
            raise
    
    def _generate_chunk_hash(self, chunk_data: Dict) -> str:
        """
        Generate unique hash for a chunk to enable idempotency
        
        Args:
            chunk_data: Chunk dictionary with text and metadata
            
        Returns:
            MD5 hash string
        """
        # Create hash from source file + chunk text (unique identifier)
        content = f"{chunk_data['source']}_{chunk_data['text'][:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _chunk_already_exists(self, chunk_hash: str) -> bool:
        """
        Check if chunk already exists in Qdrant
        
        Args:
            chunk_hash: Hash of the chunk
            
        Returns:
            True if exists, False otherwise
        """
        try:
            # Search by chunk_hash in payload
            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_hash",
                            match=MatchValue(value=chunk_hash)
                        )
                    ]
                ),
                limit=1
            )
            
            return len(results[0]) > 0
            
        except Exception as e:
            logger.warning(f"Error checking chunk existence: {e}")
            return False
    
    def find_pdf_files(self) -> List[Path]:
        """
        Find all PDF files in data directory
        
        Returns:
            List of PDF file paths
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.data_dir}")
        
        if not pdf_files:
            logger.warning("No PDF files found! Please add MSDS PDFs to the data directory.")
        
        return pdf_files
    
    def process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed chunks with metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            chunks = self.pdf_processor.process_pdf(
                str(pdf_path),
                pdf_path.name
            )
            
            if not chunks:
                logger.warning(f"No chunks extracted from {pdf_path.name}")
                return []
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path.name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return []
    
    def upload_to_qdrant(self, chunks: List[Dict]) -> int:
        """
        Upload chunks to Qdrant with idempotency
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks uploaded (excluding duplicates)
        """
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        # Prepare points for upload
        points = []
        skipped = 0
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate hash for idempotency
            chunk_hash = self._generate_chunk_hash(chunk)
            
            # Check if already exists
            if self._chunk_already_exists(chunk_hash):
                logger.debug(f"Skipping duplicate chunk: {chunk['chunk_id']}")
                skipped += 1
                continue
            
            # Create point
            point = PointStruct(
                id=chunk_hash,  # Use hash as ID for idempotency
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "chemical": chunk["chemical"],
                    "doc_type": chunk["doc_type"],
                    "severity": chunk["hazard_level"],  # Map to expected field name
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "chunk_hash": chunk_hash,
                    "timestamp": time.time(),
                    "factory_zone": "GENERAL",  # Default zone, can be customized
                    "chunk_id": chunk["chunk_id"],
                    "ingestion_date": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        if points:
            logger.info(f"Uploading {len(points)} new chunks to Qdrant...")
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"✓ Upload complete! ({skipped} duplicates skipped)")
        else:
            logger.info("No new chunks to upload (all duplicates)")
        
        return len(points)
    
    def run(self) -> Dict[str, int]:
        """
        Run complete ingestion pipeline
        
        Returns:
            Statistics dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("CHEMICAL KNOWLEDGE INGESTION PIPELINE")
        logger.info("="*60 + "\n")
        
        start_time = time.time()
        stats = {
            "total_pdfs": 0,
            "total_chunks": 0,
            "uploaded_chunks": 0,
            "failed_pdfs": 0
        }
        
        # Find PDF files
        pdf_files = self.find_pdf_files()
        stats["total_pdfs"] = len(pdf_files)
        
        if not pdf_files:
            logger.warning("No PDFs to process. Exiting.")
            return stats
        
        # Process each PDF
        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self.process_single_pdf(pdf_path)
            
            if chunks:
                all_chunks.extend(chunks)
                stats["total_chunks"] += len(chunks)
            else:
                stats["failed_pdfs"] += 1
        
        # Upload all chunks
        if all_chunks:
            uploaded = self.upload_to_qdrant(all_chunks)
            stats["uploaded_chunks"] = uploaded
        
        # Print summary
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("INGESTION SUMMARY")
        logger.info("="*60)
        logger.info(f"PDFs processed:    {stats['total_pdfs']}")
        logger.info(f"PDFs failed:       {stats['failed_pdfs']}")
        logger.info(f"Total chunks:      {stats['total_chunks']}")
        logger.info(f"New chunks added:  {stats['uploaded_chunks']}")
        logger.info(f"Time elapsed:      {elapsed:.2f}s")
        logger.info("="*60 + "\n")
        
        return stats
    
    def verify_ingestion(self, sample_size: int = 5):
        """
        Verify data was ingested correctly
        
        Args:
            sample_size: Number of random samples to display
        """
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION")
        logger.info("="*60 + "\n")
        
        # Get collection info
        collection_info = self.qdrant.get_collection(self.collection_name)
        logger.info(f"Total points in collection: {collection_info.points_count}")
        
        # Sample some points
        logger.info(f"\nSampling {sample_size} random points:\n")
        
        results = self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=sample_size
        )
        
        for idx, point in enumerate(results[0], 1):
            logger.info(f"Sample {idx}:")
            logger.info(f"  Chemical: {point.payload.get('chemical')}")
            logger.info(f"  Doc Type: {point.payload.get('doc_type')}")
            logger.info(f"  Severity: {point.payload.get('severity')}")
            logger.info(f"  Source: {point.payload.get('source')}")
            logger.info(f"  Section: {point.payload.get('section')}")
            logger.info(f"  Text preview: {point.payload.get('text')[:100]}...")
            logger.info("")


def main():
    """Main execution function"""
    try:
        # Initialize pipeline
        pipeline = MSDSIngestionPipeline(
            data_dir="data/msds",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="chemical_docs"
        )
        
        # Run ingestion
        stats = pipeline.run()
        
        # Verify results
        if stats["uploaded_chunks"] > 0:
            pipeline.verify_ingestion(sample_size=3)
        
        # Exit with appropriate code
        if stats["failed_pdfs"] > 0:
            logger.warning(f"Completed with {stats['failed_pdfs']} failed PDFs")
            sys.exit(1)
        else:
            logger.info("Ingestion completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()