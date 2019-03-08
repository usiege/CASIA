import csv
import math
import numpy as np

import json
from datetime import datetime


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.
    Args:
        image_height:
        image_width:
    Returns:
        True if both height and width divisible by 32 and False otherwise.
    """
    return image_height % 32 == 0 and image_width % 32 == 0


def load_csv(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        points = list(reader)
        return np.array(points)


def write_csv(filepath, data):
    with open(filepath, mode='w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)


def get_degree(x, y):
    d = math.atan2(y, x)
    d = d / math.pi * 180 + 180
    return d