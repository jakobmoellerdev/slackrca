# rca/slack_client.py
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_token = 'your-slack-token'
client = WebClient(token=slack_token)

def fetch_thread_messages(channel, thread_ts):
    try:
        result = client.conversations_replies(
            channel=channel,
            ts=thread_ts
        )
        return result['messages']
    except SlackApiError as e:
        print(f"Error fetching conversation: {e}")
        return []
