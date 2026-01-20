# AWS Setup Guide for Modal + DynamoDB (OIDC Integration)

This guide walks through setting up credential-free AWS access from Modal using OIDC tokens.

## Overview

Instead of storing AWS credentials as Modal Secrets (which requires rotation and poses security risks), we use Modal's OIDC integration:

1. Modal automatically generates short-lived identity tokens for your containers
2. AWS trusts Modal's OIDC provider
3. Your code exchanges the Modal token for temporary AWS credentials
4. No long-lived secrets needed!

## Prerequisites

- AWS CLI configured with admin access
- Your Modal workspace ID (run `modal token` to see it, or use the script below)
- AWS account ID

## Step 0: Get Your Modal Workspace ID

Run this in Modal to get your workspace ID:

```python
# get_modal_info.py
import modal
import os
import json

app = modal.App("get-workspace-info")

@app.function(image=modal.Image.debian_slim().pip_install("pyjwt"))
def get_workspace_info():
    import jwt
    token = os.environ.get("MODAL_IDENTITY_TOKEN")
    if not token:
        return {"error": "No MODAL_IDENTITY_TOKEN found"}
    
    claims = jwt.decode(token, options={"verify_signature": False})
    print(json.dumps(claims, indent=2))
    return claims

@app.local_entrypoint()
def main():
    get_workspace_info.remote()
```

```bash
modal run get_modal_info.py
```

Note your `workspace_id` (looks like `ac-xxxxxxxx`).

## Step 1: Create OIDC Provider in AWS

This tells AWS to trust tokens signed by Modal.

```bash
aws iam create-open-id-connect-provider \
    --url https://oidc.modal.com \
    --client-id-list oidc.modal.com
```

Save the returned ARN, it looks like:
```
arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/oidc.modal.com
```

## Step 2: Create IAM Policy for DynamoDB

Create a file `dynamodb-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DynamoDBTableAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:BatchGetItem",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/chat-table",
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/chat-table/index/*",
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/trace-table",
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/trace-table/index/*",
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/chat-metadata-table",
        "arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT_ID:table/chat-metadata-table/index/*"
      ]
    }
  ]
}
```

Replace `YOUR_ACCOUNT_ID` and adjust region/table names as needed.

Create the policy:

```bash
aws iam create-policy \
    --policy-name modal-litlens-dynamodb-policy \
    --policy-document file://dynamodb-policy.json
```

Save the policy ARN.

## Step 3: Create IAM Role with Trust Policy

Create a file `trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/oidc.modal.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.modal.com:aud": "oidc.modal.com"
        },
        "StringLike": {
          "oidc.modal.com:sub": "modal:workspace_id:YOUR_WORKSPACE_ID:*"
        }
      }
    }
  ]
}
```

Replace:
- `YOUR_ACCOUNT_ID` with your AWS account ID
- `YOUR_WORKSPACE_ID` with your Modal workspace ID (e.g., `ac-xxxxxxxx`)

Create the role:

```bash
aws iam create-role \
    --role-name modal-litlens-role \
    --assume-role-policy-document file://trust-policy.json
```

Attach the policy:

```bash
aws iam attach-role-policy \
    --role-name modal-litlens-role \
    --policy-arn arn:aws:iam::YOUR_ACCOUNT_ID:policy/modal-litlens-dynamodb-policy
```

Note the role ARN:
```
arn:aws:iam::YOUR_ACCOUNT_ID:role/modal-litlens-role
```

## Step 4: Update Your Modal Code

Now update the `rag_backend_dynamo.py` to use OIDC:

```python
# In your Modal app
import os

# Store role ARN as environment variable or in config
AWS_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/modal-litlens-role"
AWS_REGION = "us-east-2"

@modal.enter()
async def startup(self):
    # ... other initialization ...
    
    if use_dynamodb:
        from dynamo_persistence_oidc import DynamoPersistence
        
        # This automatically uses MODAL_IDENTITY_TOKEN
        self.persistence = DynamoPersistence.from_oidc(
            role_arn=AWS_ROLE_ARN,
            region=AWS_REGION,
        )
```

## Step 5: Test the Integration

Create a test script:

```python
# test_dynamo_oidc.py
import modal
import os
import json

app = modal.App("test-dynamo-oidc")

image = modal.Image.debian_slim().pip_install("boto3")

AWS_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/modal-litlens-role"
AWS_REGION = "us-east-2"

@app.function(image=image)
def test_dynamodb_access():
    import boto3
    
    # Get Modal's OIDC token
    token = os.environ.get("MODAL_IDENTITY_TOKEN")
    if not token:
        return {"error": "No MODAL_IDENTITY_TOKEN"}
    
    # Exchange for AWS credentials
    sts = boto3.client("sts", region_name=AWS_REGION)
    response = sts.assume_role_with_web_identity(
        RoleArn=AWS_ROLE_ARN,
        RoleSessionName="test-session",
        WebIdentityToken=token,
    )
    
    creds = response["Credentials"]
    print(f"✓ Got temporary AWS credentials")
    print(f"  AccessKeyId: {creds['AccessKeyId'][:10]}...")
    print(f"  Expiration: {creds['Expiration']}")
    
    # Create DynamoDB client with temp creds
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )
    
    # Try to describe a table
    try:
        table = dynamodb.Table("chat-table")
        print(f"✓ Connected to table: {table.table_name}")
        print(f"  Item count: {table.item_count}")
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    result = test_dynamodb_access.remote()
    print(json.dumps(result, indent=2, default=str))
```

```bash
modal run test_dynamo_oidc.py
```

## Restricting Access Further

You can make the trust policy more restrictive:

### Limit to specific environment:
```json
"oidc.modal.com:sub": "modal:workspace_id:ac-xxxxx:environment_name:main:*"
```

### Limit to specific app:
```json
"oidc.modal.com:sub": "modal:workspace_id:ac-xxxxx:environment_name:main:app_name:rag-backend-service:*"
```

### Limit to specific function:
```json
"oidc.modal.com:sub": "modal:workspace_id:ac-xxxxx:environment_name:main:app_name:rag-backend-service:function_name:RAGService:*"
```

## Troubleshooting

### "MODAL_IDENTITY_TOKEN not found"
- This code must run inside a Modal container, not locally
- Make sure your Modal app name follows the naming constraints (alphanumeric, dashes, periods, underscores only)

### "Access Denied" from STS
- Check that the workspace_id in your trust policy matches your actual workspace
- Verify the OIDC provider was created correctly
- Check that the policy is attached to the role

### "Table not found"
- Verify table names match
- Check the region is correct
- Ensure tables were created (run `dynamo_table_creation.py`)

## Local Development

For local development (outside Modal), use the session-based approach:

```python
import boto3

# Use SSO profile
session = boto3.Session(profile_name='AdministratorAccess-YOUR_ACCOUNT')
persistence = DynamoPersistence.from_session(session)
```

## Summary

| Item | Value |
|------|-------|
| OIDC Provider URL | `https://oidc.modal.com` |
| Client ID | `oidc.modal.com` |
| Role ARN | `arn:aws:iam::YOUR_ACCOUNT:role/modal-litlens-role` |
| Region | `us-east-2` |

The key benefit: **No AWS credentials stored anywhere!** Modal handles token generation, AWS validates the tokens, and your code gets temporary credentials that expire in 1 hour.
