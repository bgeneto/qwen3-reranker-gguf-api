from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# counters
REQUEST_COUNT = Counter("rerank_requests_total", "Total rerank requests")
FAIL_COUNT    = Counter("rerank_failures_total", "Total rerank failures")
LATENCY       = Histogram("rerank_latency_seconds", "Latency per rerank request")

def metrics_endpoint():
    def latest():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return latest
