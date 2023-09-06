#include <cstdio>
#include <unordered_map>
#include <signal.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "nccl_trace.skel.h"

static volatile sig_atomic_t exiting;
static FILE *outfile;
static bool need_comma = false;
static unsigned long event_index;

static void sig_int(int signo)
{
	exiting = 1;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

enum ncclProxyProfileState {
  ncclProxyProfileSendBegin = 0,
  ncclProxyProfileSendGPUWait = 1,
  ncclProxyProfileSendWait = 2,
  ncclProxyProfileSendEnd = 3,

  ncclProxyProfileRecvBegin = 8,
  ncclProxyProfileRecvWait = 9,
  ncclProxyProfileRecvFlushWait = 10,
  ncclProxyProfileRecvGPUWait = 11,
  ncclProxyProfileRecvEnd = 12,

  ncclProxyProfileSleep = 16,
  ncclProxyProfileWakeup = 17,

  ncclProxyProfileIdle = 24,
  ncclProxyProfileActive = 25,

  ncclProxyProfileAppend = 32,
  ncclProxyProfileAppendEnd = 33
};

static const char * collname(uint8_t coll) {
	switch (coll) {
	case 0:
		return "broadcast";
	case 1:
		return "reduce";
	case 2:
		return "allgather";
	case 3:
		return "reducescatter";
	case 4:
		return "allreduce";
	default:
		return "";
	};
}

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
	__u64 ts;
};

struct nccl_trace_key {
	uint64_t comm;
	uint64_t opCount;
	uint8_t channelId;
	uint8_t step;
	uint8_t peer;
	bool is_send;

	bool operator==(const nccl_trace_key& r) const
	{
		return (opCount == r.opCount)
			&& (comm == r.comm)
			&& (channelId == r.channelId)
			&& (step == r.step)
			&& (peer == r.peer)
			&& (is_send == r.is_send);
	}
};

struct nccl_trace_value {
	uint64_t timing[5];
	uint8_t state;
	uint8_t coll;
	uint16_t sliceSteps;
	uint16_t chunkSteps;
	uint16_t chunkSize;
	uint16_t nsteps;
	uint32_t nbytes;
};

template<>
struct std::hash<nccl_trace_key>
{
	std::size_t operator()(const nccl_trace_key& k) const noexcept {
		return std::hash<uint64_t>{}(k.opCount)
						    ^ (std::hash<uint8_t>{}(k.channelId) << 1)
						    ^ (std::hash<uint8_t>{}(k.step) << 2)
						    ^ (std::hash<uint8_t>{}(k.peer) << 3)
						    ^ (std::hash<bool>{}(k.is_send) << 4);
	}
};

static std::unordered_map<nccl_trace_key, nccl_trace_value> ops;

static void print_send(const decltype(ops)::value_type & e) {
	if (need_comma) {
		fprintf(outfile, ",\n");
	}
	need_comma = true;
	fprintf(outfile, "{\"name\": \"Send-%u-%u\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu, \"args\": {\"coll\": \"%s\", "
		"\"peer\": %u, \"step\": %u, \"opCount\": %lu, "
		"\"sliceSteps\": %u, \"chunkSteps\": %u, \"chunkSize\": %u, "
		"\"nsteps\": %u, \"nbytes\": %lu, \"comm\": \"0x%lx\"}},\n",
		(unsigned)e.first.peer, (unsigned)e.first.step, event_index,
		(unsigned)e.first.channelId, e.second.timing[0], collname(e.second.coll),
		(unsigned)e.first.peer, (unsigned)e.first.step, e.first.opCount,
		(unsigned)e.second.sliceSteps, (unsigned)e.second.chunkSteps, (unsigned)e.second.chunkSize,
		(unsigned)e.second.nsteps, (unsigned long)e.second.nbytes, e.first.comm);
	fprintf(outfile, "{\"name\": \"BufferWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[0]);
	fprintf(outfile, "{\"name\": \"BufferWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[1]);
	fprintf(outfile, "{\"name\": \"GPUWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[1]);
	fprintf(outfile, "{\"name\": \"GPUWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[2]);
	fprintf(outfile, "{\"name\": \"SendWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[2]);
	fprintf(outfile, "{\"name\": \"SendWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[3]);
	fprintf(outfile, "{\"name\": \"Send-%u-%u\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu}",
		(unsigned)e.first.peer, (unsigned)e.first.step, event_index,
		(unsigned)e.first.channelId, e.second.timing[3]);
	++event_index;
}

static void print_recv(const decltype(ops)::value_type & e) {
	if (need_comma) {
		fprintf(outfile, ",\n");
	}
	need_comma = true;
	fprintf(outfile, "{\"name\": \"Recv-%u-%u\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu, \"args\": {\"coll\": \"%s\", "
		"\"peer\": %u, \"step\": %u, \"opCount\": %lu, "
		"\"sliceSteps\": %u, \"chunkSteps\": %u, \"chunkSize\": %u, "
		"\"nsteps\": %u, \"nbytes\": %lu, \"comm\": \"0x%lx\"}},\n",
		(unsigned)e.first.peer, (unsigned)e.first.step, event_index,
		(unsigned)e.first.channelId, e.second.timing[0], collname(e.second.coll),
		(unsigned)e.first.peer, (unsigned)e.first.step, e.first.opCount,
		(unsigned)e.second.sliceSteps, (unsigned)e.second.chunkSteps, (unsigned)e.second.chunkSize,
		(unsigned)e.second.nsteps, (unsigned long)e.second.nbytes, e.first.comm);
	fprintf(outfile, "{\"name\": \"BufferWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[0]);
	fprintf(outfile, "{\"name\": \"BufferWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[1]);
	fprintf(outfile, "{\"name\": \"RecvWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[1]);
	fprintf(outfile, "{\"name\": \"RecvWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[2]);
	fprintf(outfile, "{\"name\": \"FlushWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[2]);
	fprintf(outfile, "{\"name\": \"FlushWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[3]);
	fprintf(outfile, "{\"name\": \"GPUWait\", \"cat\": \"NET\", \"ph\": \"b\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[3]);
	fprintf(outfile, "{\"name\": \"GPUWait\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu},\n",
		event_index,
		(unsigned)e.first.channelId, e.second.timing[4]);
	fprintf(outfile, "{\"name\": \"Recv-%u-%u\", \"cat\": \"NET\", \"ph\": \"e\", \"id\": %lu, "
		"\"pid\": %u, \"tid\": 1, \"ts\": %lu}",
		(unsigned)e.first.peer, (unsigned)e.first.step, event_index,
		(unsigned)e.first.channelId, e.second.timing[4]);
	++event_index;
}

static void print_event(const decltype(ops)::value_type & e) {
	if (e.second.state <= ncclProxyProfileSendEnd) {
		print_send(e);
	} else {
		print_recv(e);
	}
}

static int rb_process(void *ctx, void *data, size_t len) {
	if (len != sizeof(nccl_trace_event)) {
		return 1;
	}
	nccl_trace_event *e = reinterpret_cast<nccl_trace_event *>(data);
	const nccl_trace_key k = {e->comm, e->opCount, e->channelId, e->step, (uint8_t)e->peer, e->state <= ncclProxyProfileSendEnd};
	if ((e->state == ncclProxyProfileSendBegin)
			|| (e->state == ncclProxyProfileRecvBegin)) {
		auto pos = ops.insert({
				k,
				{ { e->ts, 0, 0, 0, 0 }, e->state, e->coll, e->sliceSteps, e->chunkSteps, e->chunkSize, e->nsteps, e->nbytes}
			});
		if (!pos.second) {
			fprintf(stderr, "Duplicate event key detected: %llu\n", e->opCount);
			return 1;
		}
	} else if (e->state <= ncclProxyProfileRecvEnd) {
		auto pos = ops.find(k);
		if (pos == ops.end()) {
			fprintf(stderr, "Unexpected event key detected\n");
			return 1;
		}
		pos->second.state = e->state;
		pos->second.timing[e->state <= ncclProxyProfileSendEnd ? e->state : e->state - ncclProxyProfileRecvBegin] = e->ts;
		if ((e->state == ncclProxyProfileSendEnd) || (e->state == ncclProxyProfileRecvEnd)) {
			print_event(*pos);
			ops.erase(pos);
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {
	int err;

	if (argc != 3) {
		fprintf(stderr, "Usage: nccl_trace_run BINARY OUTPUT\n");
		return 1;
	}

	outfile = fopen(argv[2], "w");
	if (!outfile) {
		err = errno;
		fprintf(stderr, "Failed to open output file %s: %d %s\n", argv[2], err, strerror(err));
		return 2;
	}

	fprintf(outfile, "{\"displayTimeUnit\": \"ns\", \"traceEvents\": [\n");

	libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
	libbpf_set_print(libbpf_print_fn);

	nccl_trace_bpf *skel = nccl_trace_bpf::open_and_load();
	if (!skel) {
		fprintf(stderr, "Failed to load BPF skeleton\n");
		return 3;
	}

	struct ring_buffer *rb;
	int rb_map_fd = bpf_object__find_map_fd_by_name(skel->obj, "events");
	if (rb_map_fd < 0) {
		err = errno;
		fprintf(stderr, "Failed to find event map fd: %d %s\n", err, strerror(err));
		goto cleanup;
	}

	rb = ring_buffer__new(
		rb_map_fd, rb_process, nullptr, nullptr);
	if (!rb) {
		err = errno;
		fprintf(stderr, "Failed to allocate event ring buffer: %d %s\n", err, strerror(err));
		goto cleanup;
	}

	skel->links.usdt_trace_send = bpf_program__attach_usdt(
		skel->progs.usdt_trace_send, -1, argv[1],
		"nccl", "sendrecv", nullptr);
	if (!skel->links.usdt_trace_send) {
		err = errno;
		fprintf(stderr, "Failed to attach BPF skeleton: %d %s\n", err, strerror(err));
		goto cleanup;
	}

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		err = errno;
		fprintf(stderr, "Failed to set up signal handler: %d %s\n", err, strerror(err));
		goto cleanup;
	}

	while (!exiting) {
		ring_buffer__consume(rb);
		sleep(0);
	}

	fprintf(outfile, "]}\n");
	fclose(outfile);

cleanup:
	nccl_trace_bpf::destroy(skel);
	return err;
}
