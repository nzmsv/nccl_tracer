#include <linux/kconfig.h>
#include <stdbool.h>

#include <linux/compiler_attributes.h>
//#include <linux/compiler_types.h>

#include <stddef.h>
#include <linux/stddef.h>

#define __no_sanitize_or_inline
#define __no_kasan_or_inline

#include <linux/compiler_types.h>
#include <linux/ptrace.h>
#include <linux/types.h>
#include <linux/bpf.h>

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/usdt.bpf.h>

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 4096 * 64);
} events SEC(".maps");

struct nccl_trace_event {
	__u64 comm;
	__u64 opCount;
	__u8 state;
	__u8 coll;
	__u8 step;
	__u8 channelId;
	__u16 sliceSteps;
	__u16 chunkSteps;
	__u16 chunkSize;
	__u16 nsteps;
	__u32 nbytes;
	int peer;
};

struct nccl_trace_event_ts {
	struct nccl_trace_event e;
	__u64 ts;
};

SEC("usdt")
int BPF_USDT(usdt_trace_send, struct nccl_trace_event *sendrecv) {
	struct nccl_trace_event_ts *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	if (!e) {
		return 1;
	}
	bpf_probe_read_user(&e->e, sizeof(*sendrecv), sendrecv);
	e->ts = bpf_ktime_get_ns();
	bpf_ringbuf_submit(e, 0);
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
