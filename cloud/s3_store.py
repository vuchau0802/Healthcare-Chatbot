import os
import sys
import shutil
import hashlib
import zipfile
import logging
from pathlib import Path
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "vectorstore")
S3_BUCKET       = os.environ.get("AWS_S3_BUCKET", "")
S3_KEY          = os.environ.get("AWS_S3_KEY", "medichat/vectorstore.zip")
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
ZIP_PATH        = "vectorstore.zip"

def _get_s3_client():
    """Return a boto3 S3 client using env credentials."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _zip_vectorstore(src_dir: str, zip_path: str) -> None:
    """Zip the entire vectorstore directory."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in Path(src_dir).rglob("*"):
            zf.write(file, file.relative_to(src_dir))
    size_mb = Path(zip_path).stat().st_size / (1024 ** 2)
    logger.info("Zipped vectorstore → %s (%.1f MB)", zip_path, size_mb)


def _unzip_vectorstore(zip_path: str, dest_dir: str) -> None:
    """Unzip into dest_dir, replacing any existing files."""
    if Path(dest_dir).exists():
        shutil.rmtree(dest_dir)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted vectorstore → %s/", dest_dir)


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def upload_to_s3(
    src_dir:  str = VECTORSTORE_DIR,
    bucket:   str = S3_BUCKET,
    s3_key:   str = S3_KEY,
) -> bool:
    """
    Zip the local vectorstore and upload to S3.
    Called manually after running ingest.py.

    Returns True on success.
    """
    if not bucket:
        logger.error("AWS_S3_BUCKET env var is not set.")
        return False

    if not Path(src_dir).exists():
        logger.error("Vectorstore directory '%s' not found. Run ingest.py first.", src_dir)
        return False

    print(f"Zipping {src_dir}/ …")
    _zip_vectorstore(src_dir, ZIP_PATH)

    print(f"Uploading to s3://{bucket}/{s3_key} …")
    try:
        s3 = _get_s3_client()
        s3.upload_file(
            ZIP_PATH,
            bucket,
            s3_key,
            ExtraArgs={
                "Metadata": {
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "md5":         _md5_file(ZIP_PATH),
                }
            },
            Callback=_ProgressCallback(Path(ZIP_PATH).stat().st_size),
        )
        print(f"\nUploaded → s3://{bucket}/{s3_key}")
        return True

    except NoCredentialsError:
        logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        return False
    except ClientError as e:
        logger.error("S3 upload failed: %s", e)
        return False
    finally:
        Path(ZIP_PATH).unlink(missing_ok=True)


def download_from_s3(
    dest_dir: str = VECTORSTORE_DIR,
    bucket:   str = S3_BUCKET,
    s3_key:   str = S3_KEY,
) -> bool:
    """
    Download the vectorstore zip from S3 and extract locally.
    Called at app startup if local vectorstore is missing.

    Returns True on success.
    """
    if not bucket:
        logger.warning("AWS_S3_BUCKET not set — skipping S3 download.")
        return False

    print(f"Downloading vectorstore from s3://{bucket}/{s3_key} …")
    try:
        s3   = _get_s3_client()
        meta = s3.head_object(Bucket=bucket, Key=s3_key)
        size = meta["ContentLength"]

        s3.download_file(
            bucket,
            s3_key,
            ZIP_PATH,
            Callback=_ProgressCallback(size),
        )
        print()

        _unzip_vectorstore(ZIP_PATH, dest_dir)
        print(f"✓ Vectorstore ready at {dest_dir}/")
        return True

    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            logger.error("Vectorstore not found in S3. Upload it first with: python s3_store.py upload")
        else:
            logger.error("S3 download failed: %s", e)
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return False
    finally:
        Path(ZIP_PATH).unlink(missing_ok=True)


def sync_from_s3(
    dest_dir: str = VECTORSTORE_DIR,
    bucket:   str = S3_BUCKET,
    s3_key:   str = S3_KEY,
) -> bool:
    """
    Smart sync: download from S3 only if local vectorstore is missing.
    Called automatically at app startup.

    Returns True if vectorstore is available (local or downloaded).
    """
    local_exists = Path(dest_dir).exists() and any(Path(dest_dir).iterdir())

    if local_exists:
        logger.info("Vectorstore found locally at %s/ — skipping S3 download.", dest_dir)
        return True

    logger.info("Local vectorstore not found — downloading from S3 …")
    return download_from_s3(dest_dir, bucket, s3_key)


def get_s3_status(
    bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
) -> dict:
    """
    Return metadata about the S3 vectorstore object.
    Useful for checking when it was last updated.
    """
    if not bucket:
        return {"error": "AWS_S3_BUCKET not set"}
    try:
        s3   = _get_s3_client()
        meta = s3.head_object(Bucket=bucket, Key=s3_key)
        return {
            "exists":       True,
            "size_mb":      round(meta["ContentLength"] / (1024 ** 2), 2),
            "last_modified": meta["LastModified"].isoformat(),
            "metadata":     meta.get("Metadata", {}),
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return {"exists": False}
        return {"error": str(e)}

class _ProgressCallback:
    def __init__(self, total: int):
        self._total    = total
        self._seen     = 0

    def __call__(self, chunk: int):
        self._seen += chunk
        pct = self._seen / self._total * 100 if self._total else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "upload":
        success = upload_to_s3()
        sys.exit(0 if success else 1)

    elif cmd == "download":
        success = download_from_s3()
        sys.exit(0 if success else 1)

    elif cmd == "status":
        info = get_s3_status()
        if info.get("exists"):
            print(f"  S3 vectorstore found")
            print(f"  Size:          {info['size_mb']} MB")
            print(f"  Last modified: {info['last_modified']}")
            if info.get("metadata"):
                print(f"  Uploaded at:   {info['metadata'].get('uploaded_at', 'unknown')}")
        elif info.get("exists") is False:
            print("✗ No vectorstore found in S3. Run: python s3_store.py upload")
        else:
            print(f"Error: {info.get('error')}")

    else:
        print("Usage:")
        print("  python s3_store.py upload    # upload local vectorstore to S3")
        print("  python s3_store.py download  # download vectorstore from S3")
        print("  python s3_store.py status    # check S3 vectorstore status")
