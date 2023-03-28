# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a development image containing the necessary development
# dependencies for building and testing CUDA Quantum. This does not include the
# CUDA, OpenMPI and other dependencies that some simulator backends might
# require. These backends will be omitted from the build if this environment is
# used.

# Build additional tools needed for CUDA Quantum documentation generation.
FROM ubuntu:22.04 as doxygenbuild

RUN apt update \
    && apt install -y wget unzip make cmake flex bison gcc g++ python3 \
    && wget https://github.com/doxygen/doxygen/archive/9a5686aeebff882ebda518151bc5df9d757ea5f7.zip -q -O repo.zip \
    && unzip repo.zip \
    && mv doxygen* repo \
    && cmake -G "Unix Makefiles" repo \
    && cmake --build . --parallel $(nproc) --target install --config Release

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/entrypoint.sh"]

ARG toolchain
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/home SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH="${PATH}:/usr/local/bin"

ADD llvm.tar.tzst /llvm.tar.tzst
ADD .github/workflows/docker/entrypoint.sh /entrypoint.sh

COPY --from=doxygenbuild /usr/local/bin/doxygen /usr/local/bin/doxygen

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git $toolchain ninja-build cmake \
        python3 python3-pip libpython3-dev \
        libblas-dev zstd \
    && python3 -m pip install --no-cache-dir \
        lit pytest numpy sphinx==5.3.* sphinx_rtd_theme enum-tools[sphinx] \
        breathe==4.34.* myst-parser \
    && apt-get autoremove -y --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

