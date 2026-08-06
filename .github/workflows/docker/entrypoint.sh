#!/bin/sh

CONTAINER_ALREADY_STARTED="CONTAINER_ALREADY_STARTED_PLACEHOLDER"
if [ ! -e $CONTAINER_ALREADY_STARTED ]; then
    touch $CONTAINER_ALREADY_STARTED
    echo "-- Welcome to CUDA Quantum --"
    echo "Decompressing LLVM build..."
    tar --posix --use-compress-program=unzstd -C /opt -xf llvm.tar.tzst
    rm llvm.tar.tzst
fi

exec "/bin/bash"
