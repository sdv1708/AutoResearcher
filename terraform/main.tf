# main.tf
terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "autorc-docker"
  format        = "DOCKER"
}

resource "google_cloud_run_service" "api" {
  name     = "autoresearcher-api"
  location = var.region

  template {
    spec {
      service_account_name = var.service_account_email
      containers = [{
        image = "${google_artifact_registry_repository.repo.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}/${var.image_name}:$${BUILD_ID}"
        resources = { limits = { memory = "512Mi", cpu = "1" } }
      }]
    }
  }

  traffic { percent = 100, latest_revision = true }
  autogenerate_revision_name = true

  lifecycle { ignore_changes = [template[0].spec[0].containers[0].image] }
}

resource "google_cloud_run_service_iam_member" "all_users" {
  location = google_cloud_run_service.api.location
  project  = google_cloud_run_service.api.project
  service  = google_cloud_run_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}