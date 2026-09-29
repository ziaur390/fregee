# Free-tier cloud VM — Oracle Cloud Infrastructure "Always Free".
#
# The ARM Ampere A1 shape is genuinely free forever (not a trial): 4 OCPUs and 24
# GB of RAM in total across all instances in the tenancy. This file requests 1 OCPU
# and 6 GB, which is four times what the local Multipass VM gets and leaves headroom
# for a second instance.
#
# Credentials come from ~/.oci/config, not from variables in this file. Putting an
# API key fingerprint or a private key path in a .tf file is how it ends up in git
# history, and .gitignore only covers *.tfvars — it cannot cover what you typed.
#
# Apply with:
#   terraform init
#   terraform apply -var "compartment_ocid=ocid1.compartment.oc1..xxxx" \
#                   -var "ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"

terraform {
  required_version = ">= 1.5"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0"
    }
  }
}

provider "oci" {
  # tenancy_ocid, user_ocid, fingerprint, private_key_path and region all come from
  # the named profile in ~/.oci/config.
  config_file_profile = var.oci_config_profile
}

variable "oci_config_profile" {
  description = "Profile name in ~/.oci/config."
  type        = string
  default     = "DEFAULT"
}

variable "compartment_ocid" {
  description = "Compartment to create the instance in. Required — there is no sensible default and guessing one creates resources in the wrong place."
  type        = string
}

variable "region" {
  description = <<-EOT
    Region. "Always Free" resources are only free in your home region, so changing
    this to a region you are not homed in silently starts charging.
  EOT
  type        = string
  default     = "me-jeddah-1"
}

variable "ssh_public_key" {
  description = "Public key installed on the instance. Required — Oracle images have no password login."
  type        = string
}

variable "instance_name" {
  description = "Instance display name."
  type        = string
  default     = "mlserve"
}

variable "ocpus" {
  description = "OCPUs. The Always Free A1 allowance is 4 total; 1 is ample for this stack."
  type        = number
  default     = 1
  validation {
    condition     = var.ocpus >= 1 && var.ocpus <= 4
    error_message = "The Always Free Ampere A1 allowance is between 1 and 4 OCPUs across the tenancy. More than 4 starts billing."
  }
}

variable "memory_gb" {
  description = "Memory in GB. Always Free A1 allows 24 GB total; 6 leaves room for a second instance."
  type        = number
  default     = 6
  validation {
    condition     = var.memory_gb >= 1 && var.memory_gb <= 24
    error_message = "The Always Free Ampere A1 allowance is up to 24 GB across the tenancy."
  }
}

variable "ingress_ports" {
  description = "TCP ports opened at the cloud network layer. Must match the Ansible firewall role's allowed_tcp_ports, or the two firewalls disagree and the mismatch looks like a broken app."
  type        = list(number)
  default     = [22, 80, 443, 8000, 9090, 9093, 3000]
}

# --------------------------------------------------------------------------- network

data "oci_identity_availability_domains" "available" {
  compartment_id = var.compartment_ocid
}

data "oci_core_images" "ubuntu_arm" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "24.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_vcn" "mlserve" {
  compartment_id = var.compartment_ocid
  cidr_block     = "10.0.0.0/16"
  display_name   = "${var.instance_name}-vcn"
  dns_label      = "mlserve"
}

resource "oci_core_internet_gateway" "mlserve" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.mlserve.id
  display_name   = "${var.instance_name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "mlserve" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.mlserve.id
  display_name   = "${var.instance_name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.mlserve.id
  }
}

resource "oci_core_security_list" "mlserve" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.mlserve.id
  display_name   = "${var.instance_name}-sl"

  # Egress is wide open because the host needs apt, the container registry and the
  # git remote. Egress filtering is a larger change than this host needs.
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
    description = "outbound: apt, container registry, git"
  }

  dynamic "ingress_security_rules" {
    for_each = var.ingress_ports
    content {
      protocol    = "6" # TCP
      source      = "0.0.0.0/0"
      description = "managed by terraform: tcp/${ingress_security_rules.value}"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }

  # ICMP path MTU discovery. Without it, large packets are silently dropped on
  # some routes and connections hang after the handshake — a failure that looks
  # like an application bug and is not one.
  ingress_security_rules {
    protocol    = "1"
    source      = "0.0.0.0/0"
    description = "ICMP path MTU discovery"
    icmp_options {
      type = 3
      code = 4
    }
  }
}

resource "oci_core_subnet" "mlserve" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.mlserve.id
  cidr_block                 = "10.0.1.0/24"
  display_name               = "${var.instance_name}-subnet"
  dns_label                  = "mlserve"
  route_table_id             = oci_core_route_table.mlserve.id
  security_list_ids          = [oci_core_security_list.mlserve.id]
  prohibit_public_ip_on_vnic = false
}

# --------------------------------------------------------------------------- compute

module "cloud_init" {
  source              = "../../modules/cloud-init"
  hostname            = var.instance_name
  admin_user          = "ubuntu"
  ssh_authorized_keys = [var.ssh_public_key]
  install_docker      = true
}

resource "oci_core_instance" "mlserve" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.available.availability_domains[0].name
  display_name        = var.instance_name
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = var.ocpus
    memory_in_gbs = var.memory_gb
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu_arm.images[0].id
    boot_volume_size_in_gbs = 50
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.mlserve.id
    display_name     = "${var.instance_name}-vnic"
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data           = module.cloud_init.user_data
  }

  # Replacing the instance on every metadata change would be destructive, so the
  # cloud-init config is deliberately not in here as a replacement trigger.
  lifecycle {
    ignore_changes = [metadata]
  }
}
