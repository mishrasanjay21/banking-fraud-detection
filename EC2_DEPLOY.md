# Deploy on AWS EC2

This deploys the FastAPI fraud detection API on an Ubuntu EC2 instance.

## 1. Create the EC2 instance

Use these settings in AWS EC2:

- AMI: Ubuntu Server 24.04 LTS
- Instance type: `t3.small` or larger
- Storage: at least 10 GB
- Key pair: create or select an SSH key

In the instance Security Group, add these inbound rules:

| Type | Port | Source |
| --- | ---: | --- |
| SSH | 22 | My IP |
| Custom TCP | 8000 | My IP for testing, or `0.0.0.0/0` for a public API |

Prefer allowing port 8000 only from trusted IPs. Use HTTPS through a reverse proxy before production use.

## 2. Connect to EC2

From PowerShell, replace the placeholders:

```powershell
ssh -i "C:\path\to\key.pem" ubuntu@EC2_PUBLIC_IP
```

## 3. Install Docker and Git

Run these commands inside EC2:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
printf '%s\n' \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu" \
  "$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker "$USER"
newgrp docker
```

Verify Docker:

```bash
docker --version
docker run --rm hello-world
```

## 4. Clone and run the API

```bash
git clone https://github.com/mishrasanjay21/banking-fraud-detection.git
cd banking-fraud-detection
docker build -t banking-fraud-detection .
docker run -d \
  --name banking-fraud-api \
  --restart unless-stopped \
  -p 8000:8000 \
  banking-fraud-detection
```

Check the container:

```bash
docker ps
docker logs banking-fraud-api
curl http://localhost:8000/health
```

From your computer, open:

```text
http://EC2_PUBLIC_IP:8000/docs
```

## 5. Update the deployment

After a new GitHub push:

```bash
cd ~/banking-fraud-detection
git pull
docker build -t banking-fraud-detection .
docker rm -f banking-fraud-api
docker run -d \
  --name banking-fraud-api \
  --restart unless-stopped \
  -p 8000:8000 \
  banking-fraud-detection
```

The model files are included in the repository. The large CSV datasets are intentionally ignored because the API only needs the files under `models/`.
