output "service_url" {
  value = google_cloud_run_service.api.status[0].url
}