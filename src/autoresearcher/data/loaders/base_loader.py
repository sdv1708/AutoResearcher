# src/autoresearcher/data/loaders/base_loader.py
"""
Base loader class with cloud integration hooks
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """
    Base class for all document loaders

    TODO: For cloud deployment:
    - Add GCS client initialization
    - Add BigQuery client for metadata
    - Add Pub/Sub for streaming updates
    - Add Cloud Logging integration
    """

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.stats = {
            "documents_loaded": 0,
            "documents_failed": 0,
            "bytes_processed": 0,
        }

        # TODO: Initialize cloud clients
        # self.storage_client = storage.Client()
        # self.bigquery_client = bigquery.Client()
        # self.publisher = pubsub_v1.PublisherClient()

    @abstractmethod
    def load_documents(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Load documents from source"""
        pass

    def preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Common preprocessing for all documents

        TODO: Add when moving to cloud:
        - Document deduplication check
        - Language detection
        - Content filtering
        - PII detection and removal
        """
        # Add source information
        document["source"] = self.source_name

        # TODO: Add cloud-specific metadata
        # document['load_timestamp'] = datetime.utcnow().isoformat()
        # document['loader_version'] = self.get_version()
        # document['processing_region'] = os.environ.get('GOOGLE_CLOUD_REGION', 'unknown')

        # Update stats
        self.stats["documents_loaded"] += 1

        return document

    def handle_error(self, error: Exception, context: Dict[str, Any]):
        """
        Handle loading errors

        TODO: For production:
        - Send to Cloud Error Reporting
        - Write to dead letter queue
        - Alert on repeated failures
        """
        logger.error(f"Error loading document: {error}", extra=context)
        self.stats["documents_failed"] += 1

        # TODO: Send to error reporting
        # error_client.report_exception()

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return self.stats.copy()

    def publish_document(self, document: Dict[str, Any]):
        """
        Publish document for downstream processing

        TODO: Implement for cloud:
        - Publish to Pub/Sub topic
        - Trigger Cloud Functions
        - Update BigQuery metadata
        """
        # TODO: Publish to Pub/Sub
        # topic_path = self.publisher.topic_path(project_id, topic_name)
        # future = self.publisher.publish(
        #     topic_path,
        #     json.dumps(document).encode('utf-8'),
        #     source=self.source_name
        # )
        pass
