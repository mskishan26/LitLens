"""
DynamoDB Table Creation Script
==============================

Creates three tables for the RAG pipeline:

1. chat-table: Stores Q&A turns (coupled user query + assistant answer)
   - PK: ChatId (conversation identifier)
   - SK: MessageId (timestamp-prefixed for sorting)
   
2. trace-table: Stores pipeline execution traces
   - PK: ChatId
   - SK: MessageId
   
3. chat-metadata-table: Stores chat list per user (for sidebar)
   - PK: UserId
   - SK: ChatId
   - GSI: LastUsedIndex (UserId, last_message_date) for sorting by recency

Run this ONCE to set up tables. Re-running will DELETE existing data.

Usage:
    python dynamo_table_creation.py
    
    # Or with a specific profile:
    AWS_PROFILE=my-profile python dynamo_table_creation.py
"""

import boto3
import os
import sys

# Configuration
REGION = os.environ.get("AWS_REGION", "us-east-2")
PROFILE = os.environ.get("AWS_PROFILE", "AdministratorAccess-649489225731")

# Table names
CHAT_TABLE = "chat-table"
TRACE_TABLE = "trace-table"
METADATA_TABLE = "chat-metadata-table"


def get_dynamodb():
    """Get DynamoDB resource with optional profile."""
    try:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        return session.resource("dynamodb")
    except Exception as e:
        print(f"Warning: Could not use profile '{PROFILE}': {e}")
        print("Falling back to default credentials...")
        return boto3.resource("dynamodb", region_name=REGION)


def delete_table_if_exists(dynamodb, table_name: str):
    """Delete a table if it exists and wait for deletion."""
    try:
        table = dynamodb.Table(table_name)
        table.delete()
        print(f"  Deleting existing table '{table_name}'...")
        table.wait_until_not_exists()
        print(f"  ✓ Table deleted")
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        print(f"  Table '{table_name}' does not exist (OK)")
    except Exception as e:
        print(f"  Warning: Could not delete '{table_name}': {e}")


def create_chat_table(dynamodb):
    """
    Create chat-table for storing Q&A turns.
    
    Schema:
        ChatId (PK): Conversation identifier (e.g., "conv_20250117...")
        MessageId (SK): Timestamp-prefixed message ID for chronological sorting
        
    Attributes stored:
        - query: User's question
        - answer: Generated response (raw text, not pre-formatted)
        - sources: JSON array of source documents used
        - hallucination: JSON object with verification results
        - MessageId: Link to detailed trace
        - UserId, timestamp, duration_ms, status
    """
    print(f"\nCreating '{CHAT_TABLE}'...")
    delete_table_if_exists(dynamodb, CHAT_TABLE)
    
    table = dynamodb.create_table(
        TableName=CHAT_TABLE,
        KeySchema=[
            {"AttributeName": "ChatId", "KeyType": "HASH"},
            {"AttributeName": "MessageId", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "ChatId", "AttributeType": "S"},
            {"AttributeName": "MessageId", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    
    table.wait_until_exists()
    print(f" Created with PK=ChatId, SK=MessageId")
    return table


def create_trace_table(dynamodb):
    """
    Create trace-table for storing pipeline execution traces.
    
    Schema:
        ChatId (PK): Links trace to conversation
        MessageId (SK): Unique trace identifier
        
    Attributes stored:
        - MessageId: Link back to the message
        - query, UserId, started_at, finished_at, status
        - stages: JSON object with all pipeline stage data
        - Individual stage durations (*_duration_ms) for querying
        - Summary metrics: papers_retrieved, chunks_retrieved, etc.
    """
    print(f"\nCreating '{TRACE_TABLE}'...")
    delete_table_if_exists(dynamodb, TRACE_TABLE)
    
    table = dynamodb.create_table(
        TableName=TRACE_TABLE,
        KeySchema=[
            {"AttributeName": "ChatId", "KeyType": "HASH"},
            {"AttributeName": "MessageId", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "ChatId", "AttributeType": "S"},
            {"AttributeName": "MessageId", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    
    table.wait_until_exists()
    print(f" Created with PK=ChatId, SK=MessageId")
    return table


def create_metadata_table(dynamodb):
    """
    Create chat-metadata-table for user's chat list (sidebar).
    
    Schema:
        UserId (PK): User identifier
        ChatId (SK): Conversation identifier
        
    GSI (LastUsedIndex):
        UserId (PK): Same as main table
        last_message_date (SK): For sorting by recency
        
    Attributes stored:
        - title: Chat title (usually first query, truncated)
        - last_message_date: Timestamp of most recent message
        - updated_at: Last update timestamp
    """
    print(f"\nCreating '{METADATA_TABLE}' with GSI...")
    delete_table_if_exists(dynamodb, METADATA_TABLE)
    
    table = dynamodb.create_table(
        TableName=METADATA_TABLE,
        KeySchema=[
            {"AttributeName": "UserId", "KeyType": "HASH"},
            {"AttributeName": "ChatId", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "UserId", "AttributeType": "S"},
            {"AttributeName": "ChatId", "AttributeType": "S"},
            {"AttributeName": "last_message_date", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "LastUsedIndex",
                "KeySchema": [
                    {"AttributeName": "UserId", "KeyType": "HASH"},
                    {"AttributeName": "last_message_date", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    
    table.wait_until_exists()
    print(f" Created with PK=UserId, SK=ChatId")
    print(f" GSI 'LastUsedIndex' on (UserId, last_message_date)")
    return table


def main():
    print("=" * 60)
    print("DynamoDB Table Creation for RAG Pipeline")
    print("=" * 60)
    print(f"Region: {REGION}")
    print(f"Profile: {PROFILE}")
    
    # Confirmation
    if "--yes" not in sys.argv:
        print("\n  WARNING: This will DELETE existing tables and all data!")
        response = input("Type 'yes' to continue: ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    dynamodb = get_dynamodb()
    
    # Create tables
    create_chat_table(dynamodb)
    create_trace_table(dynamodb)
    create_metadata_table(dynamodb)
    
    print("\n" + "=" * 60)
    print("✓ All tables created successfully!")
    print("=" * 60)
    
    # Summary
    print("\nTable Summary:")
    print(f"  • {CHAT_TABLE}: Q&A turns (query + answer + sources + hallucination)")
    print(f"  • {TRACE_TABLE}: Pipeline traces (stages, durations, metrics)")
    print(f"  • {METADATA_TABLE}: User chat list (title, last_message_date)")
    print("\nAccess patterns:")
    print("  • Get user's chats: Query metadata-table GSI by UserId, sorted by last_message_date")
    print("  • Get chat messages: Query chat-table by ChatId, sorted by MessageId")
    print("  • Get message trace: Get from chat-table → use MessageId to get from trace-table")


if __name__ == "__main__":
    main()