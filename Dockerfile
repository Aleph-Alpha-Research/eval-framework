# Unlike scaling, a slim cuda container (without pre-packaged pytorch, etc.) is sufficient and gives us flexibility
# But note: CUDA version matches the one in scaling and dev-nodes for good compatibility
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"

# Install Python 3.12 and pip (zstd is needed for github caching)
RUN apt-get update && \
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
    pipx && \
    # docker-cli for ability to login in startup-hook.sh (used for perturbations)
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y jq docker-ce-cli && \
    #
    apt-get autoremove && apt-get clean

# Set Workdir
RUN mkdir /eval_framework
WORKDIR /eval_framework
ENV WORKDIR=/eval_framework

# Copy over files for env installation
COPY pyproject.toml poetry.lock README.md LICENSE ./

# Install poetry into its own venv so that it does not mess up with eval_framework dependencies
RUN pipx install git+https://github.com/python-poetry/poetry.git@refs/pull/10493/head
ENV PATH="${PATH}:/root/.local/bin"

# Install into project-specific venv to prevent any conflicts with system packages
ENV POETRY_VIRTUALENVS_PATH=/venv
RUN mkdir /venv && poetry config virtualenvs.path /venv && poetry env use /usr/bin/python3.12
RUN poetry install --no-root --with dev --all-extras && poetry cache --no-interaction clear --all ""

# Install flash-attention manually, it doesn't support poetry (https://github.com/python-poetry/poetry/issues/8427)
# The version can be upgraded as required by the `transformers` package
RUN poetry run pip install --no-build-isolation flash-attn==2.7.2.post1 && poetry cache clear --no-interaction --all ""

### For optimization (docker stage caching) purposes run --no-root first and then install the package itself
# (which is nearly always being modified)
COPY . .
RUN poetry install --with dev --all-extras && poetry cache clear --no-interaction --all ""
