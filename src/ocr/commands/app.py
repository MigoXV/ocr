import logging

import typer
import uvicorn
from fastapi import FastAPI

from ocr.apizi.v1 import router

cli = typer.Typer(help="OCR service command line interface.")
api_app = FastAPI(title="OCR Image Edit Service")
api_app.include_router(router)


@cli.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", envvar="OCR_HOST"),
    port: int = typer.Option(8000, "--port", envvar="OCR_PORT"),
    reload: bool = typer.Option(False, "--reload", envvar="OCR_RELOAD"),
    workers: int = typer.Option(1, "--workers", envvar="OCR_WORKERS"),
    log_level: str = typer.Option("info", "--log-level", envvar="OCR_LOG_LEVEL"),
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvicorn.run(
        "ocr.commands.app:api_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
