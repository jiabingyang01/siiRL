# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loguru import logger
import os
import sys

SIIRL_LOG_DIRCTORY = os.getenv("SIIRL_LOG_DIRECTORY", "siirl_logs")

SIIRL_LOGGING_FILENAME = os.getenv("SIIRL_LOGGING_FILENAME", "siirl")
if SIIRL_LOGGING_FILENAME != "":
    SIIRL_LOGGING_FILENAME += "_"


def set_basic_config():
    """
    This function sets the global logging format and level. It will be called when import siirl
    """

    log_level = os.environ.get("LOGURU_LEVEL", "INFO")

    logger.remove()
    logger.level("CRITICAL", color="<bold white on red>")
    # logger.level("ERROR", color="<red><bold>")
    # logger.level("WARNING", color="<yellow>")
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level.icon}</level> "
            "<level>{level: <8}</level> | "
            # Added :<magenta>{line}</magenta> to include the line number
            "<blue>{file}</blue>:<magenta>{line}</magenta>:<cyan>{function}</cyan> >> "
            "<level>{message}</level>"
        ),
        enqueue=True,
        colorize=True,
    )
    os.makedirs(SIIRL_LOG_DIRCTORY, exist_ok=True)

    # Filename strategy:
    #   - SIIRL_LOGGING_FILENAME already carries a launch-time stamp from the shell script
    #     (e.g. "openvla_lwm_libero_spatial_20260423_025820_"), so we don't need loguru's
    #     {time} placeholder. Using {time} would cause each subprocess to stamp its own
    #     start hour, producing stray empty files when workers spawn across hour boundaries.
    #   - All processes sharing one sink path is safe because enqueue=True serializes writes
    #     within each process, and POSIX O_APPEND atomically handles concurrent <4KB writes
    #     across processes.
    base_log_name = SIIRL_LOGGING_FILENAME.rstrip("_") or "siirl"
    sink_path = os.path.join(SIIRL_LOG_DIRCTORY, base_log_name + ".log")
    logger.add(
        sink=sink_path,
        level=log_level,
        rotation="500 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {module}:{function}:{line} >> {message}",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )
