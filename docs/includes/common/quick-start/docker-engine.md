!!! info "Docker* Engine Download Source"
    Intel® ESQ uses download.docker.com* as the default download source. To use a custom download source*, configure it with:

    ```bash
    export DOCKER_SOURCE_URL=<custom_docker_source_url>
    ```

Install Docker* Engine:

```bash
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL ${DOCKER_SOURCE_URL:-https://download.docker.com/linux/ubuntu}/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: ${DOCKER_SOURCE_URL:-https://download.docker.com/linux/ubuntu}
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add your user to the docker group:

!!! warning
    Adding your user to the `docker` group grants root-level access. This should only be done on development or test systems.

```bash
sudo usermod -aG docker $USER
```

To activate your new group membership immediately in your current terminal, run:

```bash
newgrp docker
```

This command starts a new shell session with updated group permissions, allowing you to use Docker* without logging out. You can now verify Docker* installation is successful by running the `hello-world` image:

```
docker pull hello-world
```

!!! note
    If you see a list of containers (even if empty), your user is correctly added to the `docker` group. If you get a permission error, ensure you have run `newgrp docker` in your terminal. For persistent access across all sessions, log out and log back in, or reboot your system.

!!! note
    If you cannot pull the image, your network may be unable to reach the default registry. Configure a mirror registry in the Docker* daemon configuration:

    ```bash
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
    {
        "registry-mirrors": [
            "<mirror-registry-source>"
        ]
    }
    EOF
    ```

!!! info "Additional Reference"
    For detailed Docker* installation instructions, see the official [Docker* Engine installation documentation](https://docs.docker.com/engine/install).
