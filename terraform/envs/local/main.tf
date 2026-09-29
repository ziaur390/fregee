# Local development VM — Multipass.
#
# Free, offline, and works on Windows as well as macOS and Linux. This is the
# environment the Ansible playbook is meant to be developed against: a full apply
# against localhost would reconfigure your own machine, and a cloud VM costs
# either money or a signup.
#
# !! THE MULTIPASS PROVIDER IS THIRD-PARTY AND UNVERIFIED HERE !!
#
# `larstobi/multipass` is not HashiCorp-maintained, and this file could not be
# validated with `terraform validate` because Terraform is not installed on the
# machine where the rest of this repository was built and tested. The version
# constraint is a range rather than an exact pin for that reason — pinning to a
# version that does not exist in the registry breaks `terraform init` with a
# confusing error.
#
# If `terraform init` fails on the provider, the equivalent CLI is four commands
# and needs no plugins at all:
#
#   multipass launch 24.04 --name mlserve --cpus 2 --memory 2G --disk 10G
#   multipass exec mlserve -- cloud-init status --wait
#   IP=$(multipass info mlserve --format csv | tail -1 | cut -d, -f3)
#   ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ansible/inventory.ini \
#     ansible/playbook.yml -e "ansible_host=$IP"
#
# That path is what the free-tier environment also hands off to, so the Ansible
# side of this is exercised either way.

terraform {
  required_version = ">= 1.5"
  required_providers {
    multipass = {
      source  = "larstobi/multipass"
      version = ">= 1.4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.4"
    }
  }
}

provider "multipass" {}

variable "vm_name" {
  description = "Multipass instance name."
  type        = string
  default     = "mlserve"
}

variable "cpus" {
  description = <<-EOT
    vCPUs. The benchmark pins torch to one thread so the runtimes are comparable,
    so 2 is enough for the application plus the monitoring stack.
  EOT
  type        = number
  default     = 2
}

variable "memory" {
  description = <<-EOT
    RAM. The API unit sets MemoryMax=1G and the monitoring stack needs roughly
    another 700 MB. 2G is the floor at which this runs without the kernel OOM
    killer picking a victim for you.
  EOT
  type        = number
  default     = 2
}

variable "disk" {
  description = "Disk size. The torch CPU wheel plus onnxruntime is about 700 MB before the model and the image layers."
  type        = string
  default     = "10G"
}

variable "image" {
  description = "Ubuntu image. 24.04 LTS: the systemd version has ProtectProc and the other hardening directives the unit files use."
  type        = string
  default     = "24.04"
}

variable "ssh_authorized_keys" {
  description = "Public keys for the admin user. Populate this or you cannot reach the VM."
  type        = list(string)
  default     = []
}

variable "mlserve_repo" {
  description = "Repository the Ansible deploy role checks out."
  type        = string
  default     = "https://github.com/ziaur390/fregee.git"
}

variable "mlserve_version" {
  description = "Git ref to deploy."
  type        = string
  default     = "main"
}

module "cloud_init" {
  source              = "../../modules/cloud-init"
  hostname            = var.vm_name
  admin_user          = "ubuntu"
  ssh_authorized_keys = var.ssh_authorized_keys
  install_docker      = true
}

# Multipass takes the cloud-init config as a file path, not an inline string, so
# it is rendered to disk first. Ansible output, not source: gitignored.
resource "local_file" "cloud_init" {
  filename = "${path.module}/cloud-init.yaml"
  content  = module.cloud_init.user_data_raw
}

resource "multipass_instance" "mlserve" {
  name   = var.vm_name
  cpus   = var.cpus
  memory = "${var.memory}G"
  disk   = var.disk
  image  = var.image

  cloudinit_file = local_file.cloud_init.filename

  depends_on = [local_file.cloud_init]
}
