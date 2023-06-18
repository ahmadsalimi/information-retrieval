import logging

logging.basicConfig(format='%(asctime)-15s %(message)s')

level = logging.INFO


def getLogger(name):
    result = logging.getLogger(name)
    result.setLevel(level)
    return result
