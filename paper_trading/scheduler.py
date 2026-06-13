"""
Paper Trading Scheduler.
Runs the paper trading engine on a schedule during market hours.

Usage:
  python paper_trading/scheduler.py

Runs every hour at :00 minutes, Mon-Fri.
Can be left running as a background process or set up as a
Windows Task Scheduler / cron job.
"""

import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from paper_trading.engine import run_paper_trading_cycle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('paper_trading.log'),
    ]
)
logger = logging.getLogger(__name__)


def main():
    scheduler = BlockingScheduler(timezone="America/New_York")

    # Run at the top of every hour, Mon-Fri, during market hours
    # 9:00 AM to 4:00 PM ET — engine itself checks if market is open
    scheduler.add_job(
        func    = run_paper_trading_cycle,
        trigger = CronTrigger(
            day_of_week = "mon-fri",
            hour        = "9-16",
            minute      = "0",
            timezone    = "America/New_York",
        ),
        id              = "paper_trading_cycle",
        name            = "Paper Trading Hourly Cycle",
        replace_existing = True,
    )

    logger.info("=" * 60)
    logger.info("Paper Trading Scheduler started")
    logger.info("Runs every hour 9 AM - 4 PM ET, Mon-Fri")
    logger.info("Logs written to paper_trading.log")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Run once immediately on startup
    logger.info("Running initial cycle on startup...")
    run_paper_trading_cycle()

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
