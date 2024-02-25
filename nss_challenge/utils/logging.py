"""Logging utilities."""

import logging


def get_logger(name: str = "NSS Eval") -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def format_table(results, name):
    """Format the results dictionary into a table."""
    rows = []
    cols = []
    for key in results.keys():
        row, col = key.split('/')
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)
    
    # Print table header
    log_str = "{:<40}".format(name) + "".join(["{:<15}".format(col) for col in cols])
    line = "-" * len(log_str)
    log_str += "\n"
    log_str += line + "\n"

    # Print table rows
    for row in rows:
        log_str += "{:<40}".format(row)
        for col in cols:
            key = f"{row}/{col}"
            log_str += "{:<15.3f}".format(results.get(key, "NaN"))
        log_str += "\n"

    return log_str