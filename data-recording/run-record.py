import subprocess
import logging
import time
import sys
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

retries = 3
exit_code_to_retry = 55


# check git status is clean
def check_git_status():
    result = subprocess.run(['git', 'status', '--porcelain'], stdout=subprocess.PIPE, text=True)
    if result.stdout:
        logging.error('Git status is not clean. Please commit or stash your changes.')
        sys.exit(1)


# log git short hash
def log_git_short_hash():
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, text=True)
    short_hash = result.stdout.strip()
    logging.info(f'Git hash: {short_hash}')


if len(sys.argv) != 4:
    print('Usage: python run-record.py <folderPath> <beginIndex> <endIndex>')
    sys.exit(1)

check_git_status()
log_git_short_hash()

folderPath = sys.argv[1]
beginIndex = int(sys.argv[2])
endIndex = int(sys.argv[3])
logging.info(f'folderPath: {folderPath}   beginIndex: {beginIndex}   endIndex: {endIndex}')

os.makedirs(folderPath, exist_ok=True)

for index in range(beginIndex, endIndex + 1):
    logging.info(f'index: {index}')

    eventFilePath = f'{folderPath}/event{index}.raw'
    rgbFolderPath = f'{folderPath}/rgb{index}'

    for attempt in range(0, retries):
        try:
            result = subprocess.run(['python3', 'record-one.py', eventFilePath, rgbFolderPath], check=True)
            break
        except subprocess.CalledProcessError as e:
            if e.returncode == exit_code_to_retry:
                print(e)
                logging.warning(f'retrying')
                time.sleep(2)
            else:
                raise e
    else:
        logging.error(f"Failed after {retries} retries")
        raise Exception(f"Failed after {retries} retries")
