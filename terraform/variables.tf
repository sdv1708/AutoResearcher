variable "project_id"            { type = string }
variable "region" {
  description = "GCP region for Artifact Registry & Cloud Run"
  type        = string
  default     = "us-central1"   # TODO(cloud): keep in sync with _REGION
}
variable "image_name"            { type = string }
variable "service_account_email" { type = string }