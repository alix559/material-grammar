"""Run the JSON HTTP API with FastHTML's ``serve()``.

  pixi run api
  # → http://0.0.0.0:8080  (or $PORT on Railway)
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    default_port = int(os.environ.get("PORT", "8080"))
    default_device = os.environ.get("MATGRAM_DEVICE", "cpu")
    default_max_port = int(os.environ.get("MATGRAM_MAX_PORT", "8000"))
    default_timeout = float(os.environ.get("MATGRAM_STARTUP_TIMEOUT", "180"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--max-port", type=int, default=default_max_port)
    parser.add_argument(
        "--device", default=default_device, choices=["cpu", "gpu"]
    )
    parser.add_argument("--startup-timeout", type=float, default=default_timeout)
    args = parser.parse_args()

    os.environ["MATGRAM_MAX_PORT"] = str(args.max_port)
    os.environ["MATGRAM_DEVICE"] = args.device
    os.environ["MATGRAM_STARTUP_TIMEOUT"] = str(args.startup_timeout)

    # Import after env is set so the module-level ServeManager picks it up.
    from fasthtml.common import serve

    import serve_api.app  # noqa: F401 — registers routes on app

    # Explicit appname: ``python -m serve_api`` makes FastHTML look for
    # ``__main__:app``, which does not exist (``app`` lives in serve_api.app).
    serve(
        appname="serve_api.app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
