""" GCP configuration for Vertex AI and other GCP services. """

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GCPConfig:
    """GCP configuration settings"""

    project_id: str
    location: str = "us-central1"

    # Vertex AI settings
    embedding_model_id: str = "textembedding-gecko@003"
    embedding_dimension: int = 768

    # Storage settings
    bucket_name: str = "autoresearcher-data"
    index_bucket: str = ""

    # Service account settings
    service_account_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GCPConfig":
        """Create GCPConfig from environment variables"""

        return cls(
            project_id=os.getenv("GCP_PROJECT_ID", "your-project-id"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            # embedding_model_id=os.getenv("GCP_EMBEDDING_MODEL_ID", "textembedding-gecko@003"),
            # embedding_dimension=int(os.getenv("GCP_EMBEDDING_DIMENSION", 768)),
            bucket_name=os.getenv("GCP_BUCKET_NAME", "autoresearcher-data"),
            index_bucket=os.getenv("GCP_INDEX_BUCKET", ""),
            service_account_path=os.getenv("GCP_SERVICE_ACCOUNT_PATH"),
        )

    def validate(self):
        """Validate config"""
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID env variable is required")

        # TODO: when migrating to cloud
        # - enable Vertex AI API and check the bucket existence


# Global config instance
gcp_config = GCPConfig.from_env()
