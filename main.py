# main.py
import logging
from fastapi import FastAPI, HTTPException
from rca import analyze_thread, generate_pdf, fetch_thread_messages, Thread, Message

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/analyze-thread/")
async def analyze_thread_endpoint(thread: Thread):
    messages = [msg.model_dump() for msg in thread.messages]

    if not messages:
        # Fetch messages from Slack if none provided
        messages = fetch_thread_messages(thread.channel_id, thread.thread_ts)
        if not messages:
            raise HTTPException(status_code=404, detail="Thread not found or empty")

    summary = analyze_thread(messages)
    filename = 'RCA_Summary.pdf'
    generate_pdf(summary, filename)

    # For demonstration, let's pretend we're uploading it
    # In actual implementation, handle file upload to Slack or elsewhere
    return {"summary": summary, "filename": filename}

if __name__ == "__main__":
    import uvicorn
    from fastapi.testclient import TestClient

    # Initialize test client
    client = TestClient(app)

    # Mock messages for testing
    mock_messages = [
        {"text": "We encountered an error with the database connection."},
        {"text": "The issue seems to be related to network latency."},
        {"text": "Further investigation revealed a misconfiguration in the firewall settings."}
    ]

    # Define the test thread
    mock_thread = Thread(
        channel_id="mock-channel-id",
        thread_ts="mock-thread-ts",
        messages=[Message(**msg) for msg in mock_messages]
    )

    # Test the endpoint
    response = client.post("/analyze-thread/", json=mock_thread.model_dump())
    assert response.status_code == 200
    print(response.json())

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
