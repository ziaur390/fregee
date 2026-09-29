output "instance_id" {
  description = "OCID of the created instance."
  value       = oci_core_instance.mlserve.id
}

output "public_ip" {
  description = "Public IP address."
  value       = oci_core_instance.mlserve.public_ip
}

output "inventory_line" {
  description = "Paste this into the [mlserve] group in ansible/inventory.ini."
  value       = "mlserve-vm ansible_host=${oci_core_instance.mlserve.public_ip}"
}

output "next_command" {
  description = "The hand-off. Terraform creates the host; Ansible configures it."
  value       = "ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ansible/inventory.ini ansible/playbook.yml"
}

output "verify_commands" {
  description = "Confirm the service is serving and the firewall agrees with the cloud security list."
  value = [
    "curl -fsS http://${oci_core_instance.mlserve.public_ip}:8000/readyz",
    "curl -fsS http://${oci_core_instance.mlserve.public_ip}:8000/metrics | head",
    "ssh ubuntu@${oci_core_instance.mlserve.public_ip} 'sudo ufw status verbose'",
  ]
}

output "cost_warning" {
  description = "Read this."
  value = join(" ", [
    "Always Free applies to the Ampere A1 shape in your HOME region only.",
    "Region in use: ${var.region}.",
    "If that is not your home region, or if ocpus > 4 or memory_gb > 24 across the",
    "tenancy, this instance bills.",
  ])
}
