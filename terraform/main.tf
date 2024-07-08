terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "5.34.0"
    }
  }
}

provider "google" {
  credentials = local.credentials
  project = var.project
  region  = var.region
}

resource "google_storage_bucket" "mlflow_bucket" {
  name          = var.gcs_mlflow_bucket_name
  location      = var.location

  storage_class = var.gcs_storage_class
  uniform_bucket_level_access = true

  versioning {
    enabled     = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30  // days
    }
  }

  force_destroy = true
}

resource "google_storage_bucket" "raw_data_bucket" {
  name          = var.gcs_raw_data_bucket_name
  location      = var.location

  storage_class = var.gcs_storage_class
  uniform_bucket_level_access = true

  versioning {
    enabled     = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30  // days
    }
  }

  force_destroy = true
}

resource "google_compute_instance" "fraud_modelling_vm" {
  name         = "fraud-modelling-vm"
  machine_type = "e2-small"
  zone         = var.region

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/c0-deeplearning-common-cpu-v20240613-debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP
    }
  }

  tags = ["mlflow"]

  metadata_startup_script = <<-EOT
    #!/bin/bash
    echo "Startup script running..." >> /var/log/startup-script.log 2>&1
    apt-get update >> /var/log/startup-script.log 2>&1
    apt-get install -y python3-pip >> /var/log/startup-script.log 2>&1
    pip3 install mlflow >> /var/log/startup-script.log 2>&1
    mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri ${var.mlflow_backend_store_uri} --default-artifact-root ${var.mlflow_artifact_location} >> /var/log/startup-script.log 2>&1
    echo "MLflow server started" >> /var/log/startup-script.log 2>&1
    EOT
}