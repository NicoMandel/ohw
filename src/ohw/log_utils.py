import os.path

LOG_NAME = ".log.txt"

def _logpath(dirp : str) -> str:
    return os.path.join(dirp, LOG_NAME)

def append_to_log(dirp : str, img_id : str) -> None:
    fpath = _logpath(dirp)
    with open(fpath, "a") as f:
        f.write(img_id + "\n")
    return None

def log_exists(dirp : str) -> bool:
    return os.path.exists(_logpath(dirp))

def read_log(dirp : str) -> list:
    f = _logpath(dirp)
    with open(f, 'r') as file:
        content = [line.rstrip() for line in file]
    return content

def get_missing_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    not_touched = img_set - log_set
    return list(not_touched)

def get_processed_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    procesed = img_set.intersection(log_set)
    return list(procesed)

def get_unknown_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    unknown = log_set - img_set
    return unknown

def fnames_to_fids(path_list : list) -> list:
    return [os.path.basename(image_path).split(".")[0] for image_path in path_list]

# summary section instead of log - but same structure
def _summarypath(dirp : str, model_n : str) -> str:
    return os.path.join(dirp, "{}_summary.txt".format(model_n))

def summary_exists(dirp : str, model_n : str) -> bool:
    return os.path.exists(_summarypath(dirp, model_n))

def append_to_summary(dirp : str, model_n : str, img_id : str, detections : int) -> None:
    spath = _summarypath(dirp, model_n)
    with open(spath, "a") as f:
        f.write("{}, {}\n".format(img_id, detections))
    return None

def create_summary(dirp : str, model_n : str, img_ct : int) -> None:
    spath = _summarypath(dirp, model_n)
    with open(spath, "w") as f:
        f.write("Processing {} images with model {}".format(img_ct, model_n))
    return None

def prepend_summary(dirp : str, model_n : str, ds_len : int, det_count : int):
    spath = _summarypath(dirp, model_n)
    line = "{} of {} images with detections".format(det_count, ds_len)
    with open(spath, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    return None

def postprocess_summary(dirp : str, model_n : str, ds_len : int):
    spath = _summarypath(dirp, model_n)
    with open(spath, 'r') as f:
        det_count = len(f.readlines()) - 1
    print("Processed {} files with detections from {} of original dataset to {}".format(det_count, ds_len, dirp))
    prepend_summary(dirp, model_n, ds_len, det_count)