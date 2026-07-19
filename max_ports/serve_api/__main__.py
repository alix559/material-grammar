"""Run the JSON HTTP API with FastHTML's ``serve()``.

  pixi run api
  # → http://0.0.0.0:8080
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args()

    os.environ["MATGRAM_MAX_PORT"] = str(args.max_port)
    os.environ["MATGRAM_DEVICE"] = args.device
    os.environ["MATGRAM_STARTUP_TIMEOUT"] = str(args.startup_timeout)

    # Import after env is set so the module-level ServeManager picks it up.
    from fasthtml.common import serve

    import serve_api.app  # noqa: F401 — registers routes on app

    serve(host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
