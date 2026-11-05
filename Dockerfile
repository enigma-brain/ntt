FROM nvcr.io/nvidia/pytorch:25.11-py3
ENV DEBIAN_FRONTEND=noninteractive
# Add GitHub CLI apt repository
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  wget \
  git \
  gh \
  openssh-client \
  python3 \
  python3-pip \
  python3-dev \
  python3-venv \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgl1 \
  tree \
  rsync \
  tmux \
  zsh \
  sudo \
  && rm -rf /var/lib/apt/lists/*
# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /src/ntt

# Create non-root user matching host UID/GID
# Required: USER_ID, GROUP_ID, USERNAME must be set in .env
ARG USER_ID
ARG GROUP_ID
ARG USERNAME
RUN test -n "${USER_ID}" && test -n "${GROUP_ID}" && test -n "${USERNAME}" \
      || { echo "ERROR: USER_ID, GROUP_ID, and USERNAME must be set in .env. Run:"; \
           echo '  echo -e "USER_ID=$(id -u)\nGROUP_ID=$(id -g)\nUSERNAME=$(whoami)" >> .env'; \
           exit 1; } && \
    groupadd -g ${GROUP_ID} ${USERNAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    chown -R ${USER_ID}:${GROUP_ID} /src/ntt

USER ${USERNAME}

# Copy minimal files needed for package install
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml uv.lock* .python-version* ./
COPY --chown=${USER_ID}:${GROUP_ID} README.md ./
COPY --chown=${USER_ID}:${GROUP_ID} src/ ./src/
# Install everything including local package as the non-root user
RUN uv sync
RUN git config --global --add safe.directory /src/ntt/

# Activate uv environment by default, and put uv tool install bin on PATH
ENV PATH="/src/ntt/.venv/bin:/home/${USERNAME}/.local/bin:$PATH"
EXPOSE 8887
