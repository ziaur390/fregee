output "vm_name" {
  description = "Multipass instance name."
  value       = multipass_instance.mlserve.name
}

output "ipv4" {
  description = "Instance IP address, for the Ansible inventory."
  value       = try(multipass_instance.mlserve.ipv4, null)
}

output "inventory_line" {
  description = "Paste this into the [mlserve] group in ansible/inventory.ini."
  value       = "mlserve-vm ansible_host=${try(multipass_instance.mlserve.ipv4, "<ip>")}"
}

output "next_command" {
  description = <<-EOT
    The hand-off. Terraform creates the host and stops; Ansible configures it.
    They are not merged into one apply because configuration must be re-runnable
    without risking the machine being recreated.
  EOT
  value       = "ansible-playbook -i ansible/inventory.ini ansible/playbook.yml -e 'mlserve_repo=${var.mlserve_repo} mlserve_version=${var.mlserve_version}'"
}

output "verify_command" {
  description = "Run this after the playbook to confirm the service is serving."
  value       = "curl -fsS http://${try(multipass_instance.mlserve.ipv4, "<ip>")}:8000/readyz"
}

output "shell_command" {
  description = "Shell into the VM."
  value       = "multipass shell ${multipass_instance.mlserve.name}"
}
