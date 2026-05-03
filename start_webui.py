import uvicorn  # Import uvicorn so this helper starts the FastAPI WebUI with one stable command.

HOST = "127.0.0.1"  # Keep the development server bound to the local machine.
PORT = 8011  # Use a different port from 8000 so old cached or already-running servers cannot collide.
URL = f"http://{HOST}:{PORT}/hard-reset-webui-20260503-14"  # Build the hard-reset URL that should not exist in browser cache.


if __name__ == "__main__":  # Run the server only when this file is executed directly.
    print("=" * 72)  # Print a visible separator before the startup instructions.
    print("半自動標註工具 WebUI hard reset 啟動")  # Print the tool startup title.
    print("請開這個網址，不要開舊的 8000 或 /webui：")  # Tell the user which URL to use.
    print(URL)  # Print the exact hard-reset URL.
    print("=" * 72)  # Print a visible separator after the startup instructions.
    uvicorn.run("web_fastapi.app:app", host=HOST, port=PORT, reload=False)  # Start the FastAPI app on the clean hard-reset port.
