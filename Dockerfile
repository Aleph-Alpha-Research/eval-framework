# Unlike scaling, a slim cuda container (without pre-packaged pytorch, etc.) is sufficient and gives us flexibility
# But note: CUDA version matches the one in scaling and dev-nodes for good compatibility
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"
ENV CUDA_HOME="/usr/local/cuda"

# Remove automatic cleanup of apt cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install Python 3.12 and pip (zstd is needed for github caching)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=apt-lists \
    export DEBIAN_FRONTEND="noninteractive" && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      python3.12-dev \
      openssh-client \
      htop \
      curl \
      ca-certificates \
      ibverbs-providers \
      libibverbs1  \
      librdmacm1 \
      git \
      zstd \
      jq \
      python3-pip \
      # Correct language support
      locales && \
    # docker-cli for ability to login in startup-hook.sh (used for perturbations)
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update &&  \
    apt-get install -y jq docker-ce-cli && \
    # Needed for determined
    mkdir -p /var/run/sshd && \
    # Configure locales for UTF-8 support
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

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
COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=target=$UV_CACHE_DIR,type=cache,sharing=locked,id=uv-cache \
  uv sync --frozen --link-mode="copy" --all-extras --group cu124 --group flash-attn --no-install-project

# For better docker stage caching, we install the package separately, since that changes
# more frequently than the dependencies in the lock file.
COPY . .
RUN --mount=target=$UV_CACHE_DIR,type=cache,sharing=locked,id=uv-cache \
  uv sync --inexact --link-mode="copy" --all-extras --group cu124 --group flash-attn
