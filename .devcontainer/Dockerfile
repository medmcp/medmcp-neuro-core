# syntax=docker/dockerfile:1
#
# Dev container for medmcp-neuro-core — the shared medmcp-base + dev toolchain (uv is
# already in the base; add just, git, build tools, and the Docker CLI so the
# stack image can be built/tested from inside). No Node: this stack has no
# frontend. Python deps are synced at postCreate against the mounted source.
#
# No GPU device is requested by default so the container starts on any host; to
# run GPU code from inside, add `--device nvidia.com/gpu=all` (CDI) — or just build
# the image (`just docker-build`) and run that with the device.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Trust extra CA certs for use behind a TLS-intercepting (MITM) corporate proxy.
# Drop the proxy's root CA as a *.crt into ./certs/ (gitignored; empty = no-op),
# then rebuild. UV_NATIVE_TLS makes uv use the system trust store so it honours
# them (uv's bundled roots otherwise reject the proxy's re-signed certificates).
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
# uv uses the system store via UV_NATIVE_TLS; Python's requests/urllib (HD-BET,
# FastSurfer downloaders) trust certifi, so point them at the system bundle too.
ENV UV_NATIVE_TLS=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN curl --proto '=https' --tlsv1.2 -fsSL https://just.systems/install.sh \
    | bash -s -- --to /usr/local/bin
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /workspaces/medmcp-neuro-core
