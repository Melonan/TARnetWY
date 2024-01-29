import os
import logging
from datetime import datetime

def setup_logging(dir):
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    log_directory = os.path.join(dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)

    # 获取当前日期
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 构建日志文件路径
    log_file_path = os.path.join(log_directory, f'fib_detec_{current_date}.log')

    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 创建文件处理程序（输出到文件）
        fh = logging.FileHandler(log_file_path, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # 创建流处理程序（输出到控制台）
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

if __name__ == "__main__":
    pass