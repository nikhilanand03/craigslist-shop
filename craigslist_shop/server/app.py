"""
FastAPI application for the Craigslist Shop Environment.

This module creates an HTTP server that exposes the CraigslistShopEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CraigslistShopAction, CraigslistShopObservation
    from .craigslist_shop_environment import CraigslistShopEnvironment
except (ModuleNotFoundError, ImportError):
    from models import CraigslistShopAction, CraigslistShopObservation
    from server.craigslist_shop_environment import CraigslistShopEnvironment


# Create the app with web interface and README integration
app = create_app(
    CraigslistShopEnvironment,
    CraigslistShopAction,
    CraigslistShopObservation,
    env_name="craigslist_shop",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn burgershack.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
