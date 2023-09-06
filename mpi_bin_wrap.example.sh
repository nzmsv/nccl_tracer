#!/usr/bin/env bash
#
# Wrapper to handle any rank-specific customization
#
# Usage: mpirun (...) mpi_bin_wrap.sh my_app app_args
#

set -e

NCCL_SO="/usr/local/lib/libnccl.2.so"

sudo ${PWD}/nccl_trace "$NCCL_SO" "/tmp/nccl_proxy_trace_rank${OMPI_COMM_WORLD_RANK}.json" &
BPFPID=$!

"$@"

sleep 5
setsid sudo kill -INT $BPFPID
wait
