# Unlike scaling, a slim cuda container (without pre-packaged pytorch, etc.) is sufficient and gives us flexibility
# But note: CUDA version matches the one in scaling and dev-nodes for good compatibility
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"
ENV CUDA_HOME="/usr/local/cuda"

# Remove automatic cleanup of apt cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install system dependencies  pip (zstd is needed for github caching)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=apt-lists \
    export DEBIAN_FRONTEND="noninteractive" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates \
      curl && \
    # docker-cli for ability to login in startup-hook.sh (used for perturbations)
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    # Installation
    apt-get update && \
    apt-get install -y --no-install-recommends \
      htop \
      ibverbs-providers \
      libibverbs1  \
      librdmacm1 \
      git \
      jq \
      docker-ce-cli \
      # determined
      openssh-client \
      openssh-server \
      # Needed for Github caching
      zstd \
      # Correct language support
      locales && \
    # Needed for determined
    mkdir -p /var/run/sshd && \
    # Configure locales for UTF-8 support
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# Configure environment variables for uv
RUN mkdir -p /uv/{venv,cache,python}
ENV UV_PROJECT_ENVIRONMENT='/uv/venv' \
    UV_PYTHON_BIN_DIR="/uv/python" \
    VIRTUAL_ENV='/uv/venv' \
    UV_CACHE_DIR="/uv/cache"
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}:${UV_PYTHON_BIN_DIR}"

WORKDIR /eval_framework
ENV WORKDIR=/eval_framework

# Install pre-commit
RUN uv tool install --no-cache pre-commit

# Copy over files for env installation
COPY pyproject.toml uv.lock README.md LICENSE .python-version ./

RUN --mount=target=$UV_CACHE_DIR,type=cache,sharing=locked,id=uv-cache \
  uv sync --frozen --link-mode="copy" --all-extras --group flash-attn --no-install-project

# For better docker stage caching, we install the package separately, since that changes
# more frequently than the dependencies in the lock file.
COPY . .
RUN --mount=target=$UV_CACHE_DIR,type=cache,sharing=locked,id=uv-cache \
  uv sync --link-mode="copy" --all-extras --group flash-attn
