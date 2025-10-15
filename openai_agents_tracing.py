from typing import Any
from agents import Span, Trace, TracingProcessor


class NoOpTracingProcessor(TracingProcessor):
    def on_trace_start(self, trace: "Trace") -> None:
        pass
    
    def on_trace_end(self, trace: "Trace") -> None:
        pass

    def on_span_start(self, span: "Span[Any]") -> None:
        pass

    def on_span_end(self, span: "Span[Any]") -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass