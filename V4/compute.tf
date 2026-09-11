# --- AMI LOOKUP (Ubuntu 22.04 LTS) ---
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical official ID

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- DEFAULT VPC LOOKUP ---
data "aws_vpc" "default" {
  default = true
}

# --- PUBLIC SSH KEY RESOURCE ---
resource "aws_key_pair" "ao3_key" {
  key_name   = "ao3-scraper-key"
  public_key = var.public_key
}

# --- CUSTOM SECURITY GROUP ---
resource "aws_security_group" "ao3_scraper_sg" {
  name        = "ao3-scraper-playground-sg"
  description = "Allow inbound SSH access for AO3 research scraper"
  vpc_id      = data.aws_vpc.default.id

  # Inbound SSH rule (Required for remote execution and SCP)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound traffic (Allows scraper to query AO3 over HTTPS/HTTP)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ao3-scraper-sg"
  }
}

# --- EC2 INSTANCE ---
resource "aws_instance" "ao3_scraper" {
  ami                    = data.aws_ami.ubuntu.id 
  instance_type          = "t3.micro" # Strictly within KodeKloud free limits
  key_name               = aws_key_pair.ao3_key.key_name
  vpc_security_group_ids = [aws_security_group.ao3_scraper_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              # Update package indexes and install required Python packages
              sudo apt-get update -y
              sudo apt-get install -y python3 python3-pip python3-pandas python3-bs4 python3-requests python3-numpy

              # Setup application directory
              mkdir -p /home/ubuntu/app
              chown -R ubuntu:ubuntu /home/ubuntu/app
              EOF

  tags = {
    Name = "ao3-research-scraper"
  }
}

# --- OUTPUTS ---
output "ssh_connection_string" {
  description = "Command to SSH into your EC2 instance from your local terminal"
  value       = "ssh -i /path/to/your-private-key ubuntu@${aws_instance.ao3_scraper.public_ip}"
}

output "scp_upload_command" {
  description = "Command to upload your Python script to the instance"
  value       = "scp -i /path/to/your-private-key ao3_research_sampler_v4.py ubuntu@${aws_instance.ao3_scraper.public_ip}:~/app/"
}

output "scp_download_command" {
  description = "Command to download generated CSV results to your local laptop"
  value       = "scp -i /path/to/your-private-key ubuntu@${aws_instance.ao3_scraper.public_ip}:~/app/ao3_results.csv ./"
}

# --- VARIABLES ---
variable "public_key" {
  type        = string
  description = "Your local SSH public key string (contents of ~/.ssh/id_rsa.pub or ~/.ssh/id_ed25519.pub)"
}