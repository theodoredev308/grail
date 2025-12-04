#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from grail.infrastructure.checkpoints import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.network import create_subtensor
from grail.infrastructure.comms import clear_client_cache
from grail.shared.constants import TRAINER_UID, NETUID


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [dl-manager] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


async def _resolve_checkpoint_credentials() -> Any:
    # Prefer trainer bucket from chain (matches default miner behavior)
    use_trainer = str(os.getenv("USE_TRAINER_BUCKET", "1")).lower() in ("1", "true", "yes", "on")
    if use_trainer:
        try:
            import bittensor as bt  # type: ignore

            wallet = bt.wallet(
                name=os.getenv("BT_WALLET_COLD", "default"),
                hotkey=os.getenv("BT_WALLET_HOT", "default"),
            )
            netuid = int(os.getenv("NETUID", str(NETUID)))
            subtensor = await create_subtensor()
            metagraph = await subtensor.metagraph(netuid)
            chain = GrailChainManager(
                type("Cfg", (), {"netuid": netuid})(),
                wallet,
                metagraph,
                subtensor,
                load_r2_credentials(),
            )
            await chain.initialize()
            trainer_bucket = chain.get_bucket(TRAINER_UID)
            if trainer_bucket is not None:
                logging.info("Using trainer bucket from chain (UID %d): %s", TRAINER_UID, trainer_bucket.name.strip() if hasattr(trainer_bucket, 'name') else 'unknown')
                return trainer_bucket
            else:
                logging.warning("Trainer bucket (UID %d) not found in chain commitments; falling back to local credentials", TRAINER_UID)
        except Exception as e:
            logging.warning("Trainer bucket resolution failed: %s; falling back to local creds", e)

    try:
        creds = load_r2_credentials()
        bucket_name = creds.bucket_name if hasattr(creds, 'bucket_name') else 'unknown'
        logging.info("Using local R2 credentials (bucket: %s)", bucket_name)
        return creds.get_read_dict()
    except Exception:
        logging.warning("Failed to load R2 credentials; set GRAIL_* env")
        return {
            "name": os.getenv("GRAIL_CKPT_BUCKET", ""),
            "account_id": os.getenv("GRAIL_CKPT_ACCOUNT_ID", ""),
            "access_key_id": os.getenv("GRAIL_CKPT_ACCESS_KEY", ""),
            "secret_access_key": os.getenv("GRAIL_CKPT_SECRET_KEY", ""),
        }


async def main() -> None:
    keep_limit = int(os.getenv("GRAIL_CKPT_KEEP", "5"))  # Keep 5 windows by default
    bucket_check_interval = float(os.getenv("DL_MANAGER_CHECK_INTERVAL", "30.0"))  # Check bucket every 30 seconds

    creds = await _resolve_checkpoint_credentials()
    mgr = CheckpointManager(
        cache_root=default_checkpoint_cache_root(),
        credentials=creds,
        keep_limit=keep_limit,
    )

    downloaded_windows: set[int] = set()  # Track which windows we've downloaded
    
    logging.info("Starting download manager (keep=%d, polling bucket every %.1fs)", keep_limit, bucket_check_interval)

    async def download_checkpoint(window: int) -> bool:
        """Download a checkpoint if it's ready and not already downloaded."""
        if window in downloaded_windows:
            return True  # Already downloaded
        
        # Check if checkpoint is ready
        if not await mgr._is_checkpoint_ready(window):
            return False  # Not ready yet
        
        logging.info("Downloading new checkpoint for window %s", window)
        try:
            # Clear cache before download
            try:
                await clear_client_cache()
            except Exception:
                pass
            
            path = await mgr.get_checkpoint(window)
            if path is not None:
                downloaded_windows.add(window)
                logging.info("✅ Successfully downloaded checkpoint for window %s at %s", window, path)
                return True
            else:
                logging.warning("Checkpoint %s not ready yet (READY marker not found)", window)
                return False
        except Exception as e:
            logging.warning("Failed to download checkpoint %s: %s", window, e)
            return False

    async def check_bucket_for_new_checkpoints() -> None:
        """Check bucket for new checkpoints, download them, and cleanup old ones."""
        try:
            remote_windows = await mgr.list_remote_windows()
            if not remote_windows:
                logging.debug("No checkpoints found in bucket")
                return
            
            latest_window = max(remote_windows)
            logging.debug("Found %d remote checkpoints, latest is window %s", len(remote_windows), latest_window)
            
            # Only download recent checkpoints (within keep_limit + some buffer)
            # This prevents downloading all historical checkpoints
            from grail.shared.constants import WINDOW_LENGTH
            max_windows_to_download = keep_limit + 2  # Keep limit + 2 extra for safety
            cutoff_window = latest_window - (max_windows_to_download * WINDOW_LENGTH)
            
            # Filter to only recent windows
            recent_windows = [w for w in remote_windows if w >= cutoff_window]
            logging.debug("Filtering to %d recent windows (cutoff: %s)", len(recent_windows), cutoff_window)
            
            # Download any new recent checkpoints that are ready
            for window in sorted(recent_windows):
                if window not in downloaded_windows:
                    await download_checkpoint(window)
            
            # Cleanup old checkpoints (keep only last N windows based on latest)
            try:
                await mgr.cleanup_local(latest_window)
            except Exception as e:
                logging.debug("Cleanup error (non-fatal): %s", e)
            
        except Exception as e:
            logging.warning("Error checking bucket for new checkpoints: %s", e)

    logging.info("Starting bucket polling loop (checking every %.1fs)", bucket_check_interval)
    
    while True:
        try:
            await check_bucket_for_new_checkpoints()
            await asyncio.sleep(bucket_check_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.warning("download loop error: %s", e)
            await asyncio.sleep(2.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
