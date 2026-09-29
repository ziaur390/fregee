# Reusable cloud-init module.
#
# Only the cloud-init config is shared between environments, and that is
# deliberate. The compute resources are genuinely different — a Multipass VM and
# an Oracle Cloud instance are not interchangeable behind one interface, and a
# module that pretended otherwise would be an abstraction with two
# implementations and no second use.
#
# What IS identical is the boot contract: install Python 3, Docker and the
# monitoring prerequisites, create the service user, and hand off to Ansible. That
# is what lives here.

terraform {
  required_version = ">= 1.5"
}

variable "hostname" {
  description = "Hostname to set on the instance."
  type        = string
  default     = "mlserve"
}

variable "admin_user" {
  description = "Non-root user that Ansible connects as."
  type        = string
  default     = "ubuntu"
}

variable "ssh_authorized_keys" {
  description = "Public keys to install for the admin user. Key-only auth is enforced later by the Ansible users role."
  type        = list(string)
  default     = []
}

variable "install_docker" {
  description = <<-EOT
    Install Docker at boot. Left configurable because the Ansible docker role also
    installs it from Docker's own repository — doing it in both places is
    harmless (apt is idempotent) but doing it only here would bypass the daemon.json
    log-capping configuration, which is the part that actually matters.
  EOT
  type        = bool
  default     = true
}

variable "timezone" {
  description = "System timezone. UTC so metric timestamps and drift windows line up."
  type        = string
  default     = "UTC"
}

variable "extra_packages" {
  description = "Additional apt packages to install at boot."
  type        = list(string)
  default     = []
}

output "user_data" {
  description = "Rendered cloud-init configuration, base64-encoded as the providers expect."
  value       = base64encode(templatefile("${path.module}/cloud-init.yaml.tftpl", {
    hostname      = var.hostname
    admin_user    = var.admin_user
    ssh_keys      = var.ssh_authorized_keys
    install_docker = var.install_docker
    timezone      = var.timezone
    extra_packages = var.extra_packages
  }))
}

output "user_data_raw" {
  description = "Rendered cloud-init configuration as plain text, for inspection."
  value       = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    hostname      = var.hostname
    admin_user    = var.admin_user
    ssh_keys      = var.ssh_authorized_keys
    install_docker = var.install_docker
    timezone      = var.timezone
    extra_packages = var.extra_packages
  })
}
