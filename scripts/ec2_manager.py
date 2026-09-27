#!/usr/bin/env python3
"""
AWS EC2 Automation Manager for ML Challenge 2026.

Supports:
- Checking credentials & region
- Creating/verifying SSH Key Pair (~/.ssh/ml-challenge-ec2-key.pem)
- Creating/verifying Security Group (port 22 open)
- Launching an optimized compute instance (default: c6i.4xlarge or c6i.8xlarge)
- Syncing project files & datasets via rsync
- Running remote training & inference in tmux
- Syncing outputs back to local machine
- Terminating instance
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("boto3 is required: pip install boto3")
    sys.exit(1)

DEFAULT_KEY_NAME = "ml-challenge-ec2-key"
DEFAULT_SG_NAME = "ml-challenge-sg"
DEFAULT_INSTANCE_TYPE = "c6i.4xlarge"  # 16 vCPUs, 32 GB RAM (~$0.68/hr)
DEFAULT_VOLUME_SIZE = 100  # GB gp3


def get_default_region():
    session = boto3.session.Session()
    region = session.region_name
    if not region:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return region


def get_latest_ubuntu_ami(ec2_client):
    """Retrieve the latest official Ubuntu 24.04 LTS (Noble) x86_64 AMI."""
    try:
        response = ec2_client.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]},
            ],
        )
        images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if images:
            return images[0]["ImageId"]
    except Exception as e:
        print(f"Warning: could not lookup dynamic Ubuntu AMI: {e}")
    
    # Fallback to standard Ubuntu 22.04 LTS AMI lookup
    response = ec2_client.describe_images(
        Owners=["099720109477"],
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No suitable Ubuntu AMI found.")
    return images[0]["ImageId"]


def ensure_key_pair(ec2_client, key_name=DEFAULT_KEY_NAME):
    """Ensure the SSH key pair exists locally and on AWS."""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    pem_path = ssh_dir / f"{key_name}.pem"

    try:
        ec2_client.describe_key_pairs(KeyNames=[key_name])
        if pem_path.exists():
            pem_path.chmod(0o400)
            return str(pem_path)
        print(f"Notice: Key pair '{key_name}' exists in AWS, but '{pem_path}' is missing locally.")
        print(f"Deleting remote key pair '{key_name}' to regenerate with a matching private key...")
        ec2_client.delete_key_pair(KeyName=key_name)
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidKeyPair.NotFound":
            raise

    print(f"Creating new AWS EC2 Key Pair: {key_name}...")
    resp = ec2_client.create_key_pair(KeyName=key_name, KeyType="rsa", KeyFormat="pem")
    pem_path.write_text(resp["KeyMaterial"], encoding="utf-8")
    pem_path.chmod(0o400)
    print(f"Saved private key to: {pem_path}")
    return str(pem_path)


def ensure_security_group(ec2_client, sg_name=DEFAULT_SG_NAME):
    """Ensure a security group with SSH port 22 access exists."""
    try:
        resp = ec2_client.describe_security_groups(GroupNames=[sg_name])
        return resp["SecurityGroups"][0]["GroupId"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidGroup.NotFound":
            raise

    print(f"Creating Security Group '{sg_name}' with SSH access...")
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"] if vpcs["Vpcs"] else None

    kwargs = {"GroupName": sg_name, "Description": "ML Challenge 2026 ER Cluster SG"}
    if vpc_id:
        kwargs["VpcId"] = vpc_id

    resp = ec2_client.create_security_group(**kwargs)
    sg_id = resp["GroupId"]

    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
            }
        ],
    )
    return sg_id


def launch_instance(args):
    region = args.region or get_default_region()
    print(f"Connecting to AWS EC2 in region [{region}]...")
    session = boto3.session.Session(region_name=region)
    ec2_client = session.client("ec2")
    ec2_res = session.resource("ec2")

    # Check caller identity
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    print(f"Authenticated as AWS Principal: {identity['Arn']}")

    pem_file = ensure_key_pair(ec2_client, args.key_name)
    sg_id = ensure_security_group(ec2_client, args.sg_name)
    ami_id = args.ami or get_latest_ubuntu_ami(ec2_client)
    print(f"Using AMI: {ami_id} (Ubuntu LTS)")

    user_data = """#!/bin/bash
set -e
apt-get update
apt-get install -y python3-pip python3-venv git htop tmux unzip rsync libgomp1
"""

    print(f"Launching EC2 instance: {args.instance_type} (100GB gp3 root volume)...")
    instances = ec2_res.create_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        KeyName=args.key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": args.volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        UserData=user_data,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "ml-challenge-er-worker"}],
            }
        ],
    )

    instance = instances[0]
    print(f"Instance requested: {instance.id}. Waiting for instance to enter 'running' state...")
    instance.wait_until_running()
    instance.reload()

    public_ip = instance.public_ip_address
    print("\n=======================================================")
    print(f"EC2 Instance is RUNNING!")
    print(f"Instance ID: {instance.id}")
    print(f"Public IP:   {public_ip}")
    print(f"SSH Key:     {pem_file}")
    print("=======================================================")
    print("\nTo SSH into the instance:")
    print(f"  ssh -i {pem_file} ubuntu@{public_ip}")
    print("\nTo sync project files and dataset:")
    print(f"  python3 scripts/ec2_manager.py sync-up --ip {public_ip}")
    print("\nTo terminate when finished:")
    print(f"  python3 scripts/ec2_manager.py terminate --instance-id {instance.id}")

    # Save state to local file for convenience
    state_file = Path("scripts/.ec2_state.json")
    state_file.write_text(
        json.dumps({
            "instance_id": instance.id,
            "public_ip": public_ip,
            "region": region,
            "key_file": pem_file,
            "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2)
    )


def sync_up(args):
    state = load_state()
    ip = args.ip or (state.get("public_ip") if state else None)
    key_file = args.key or (state.get("key_file") if state else str(Path.home() / f".ssh/{DEFAULT_KEY_NAME}.pem"))

    if not ip:
        print("Error: EC2 Public IP not provided and not found in scripts/.ec2_state.json")
        sys.exit(1)

    print(f"Syncing local files to EC2 [{ip}]...")
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", f"ssh -i {key_file} -o StrictHostKeyChecking=no",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", "*.zip",
        ".",
        f"ubuntu@{ip}:~/ml-challenge/",
    ]
    subprocess.run(cmd, check=True)
    print("Sync completed!")


def sync_down(args):
    state = load_state()
    ip = args.ip or (state.get("public_ip") if state else None)
    key_file = args.key or (state.get("key_file") if state else str(Path.home() / f".ssh/{DEFAULT_KEY_NAME}.pem"))

    if not ip:
        print("Error: EC2 Public IP not provided and not found in scripts/.ec2_state.json")
        sys.exit(1)

    print(f"Syncing output/ from EC2 [{ip}] to local...")
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", f"ssh -i {key_file} -o StrictHostKeyChecking=no",
        f"ubuntu@{ip}:~/ml-challenge/output/",
        "output/",
    ]
    subprocess.run(cmd, check=True)
    print("Sync down completed!")


def terminate_instance(args):
    state = load_state()
    inst_id = args.instance_id or (state.get("instance_id") if state else None)
    region = args.region or (state.get("region") if state else get_default_region())

    if not inst_id:
        print("Error: Instance ID not provided and not found in scripts/.ec2_state.json")
        sys.exit(1)

    session = boto3.session.Session(region_name=region)
    ec2_client = session.client("ec2")
    print(f"Terminating instance {inst_id} in {region}...")
    ec2_client.terminate_instances(InstanceIds=[inst_id])
    print("Termination signal sent.")
    state_file = Path("scripts/.ec2_state.json")
    if state_file.exists():
        state_file.unlink()


def load_state():
    state_file = Path("scripts/.ec2_state.json")
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="AWS EC2 ML Challenge Pipeline Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Launch
    p_launch = subparsers.add_parser("launch", help="Launch a new EC2 instance")
    p_launch.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help=f"Default: {DEFAULT_INSTANCE_TYPE}")
    p_launch.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE, help="Root EBS size (GB)")
    p_launch.add_argument("--region", default=None, help="AWS region (e.g. us-east-1, ap-south-1)")
    p_launch.add_argument("--key-name", default=DEFAULT_KEY_NAME)
    p_launch.add_argument("--sg-name", default=DEFAULT_SG_NAME)
    p_launch.add_argument("--ami", default=None, help="Custom AMI ID")

    # Sync Up
    p_sync_up = subparsers.add_parser("sync-up", help="Rsync code and dataset up to EC2")
    p_sync_up.add_argument("--ip", default=None)
    p_sync_up.add_argument("--key", default=None)

    # Sync Down
    p_sync_down = subparsers.add_parser("sync-down", help="Rsync output results down from EC2")
    p_sync_down.add_argument("--ip", default=None)
    p_sync_down.add_argument("--key", default=None)

    # Terminate
    p_term = subparsers.add_parser("terminate", help="Terminate EC2 instance")
    p_term.add_argument("--instance-id", default=None)
    p_term.add_argument("--region", default=None)

    args = parser.parse_args()

    if args.command == "launch":
        launch_instance(args)
    elif args.command == "sync-up":
        sync_up(args)
    elif args.command == "sync-down":
        sync_down(args)
    elif args.command == "terminate":
        terminate_instance(args)


if __name__ == "__main__":
    main()
