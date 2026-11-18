import os
from pySmartDL import SmartDL
from tqdm import tqdm


save_dir = r'F:\AI\Dataset\OpenBMI'

retry = 5

for session in range(1, 3):
    if session < 10:
        session_add_zero = '0' + str(session)
    else:
        session_add_zero = str(session)
    for subject in tqdm(range(1, 55)):
        if subject < 10:
            subject_add_zero = '0' + str(subject)
        else:
            subject_add_zero = str(subject)
        url = ('https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub'
               '/10.5524/100001_101000/100542/'
               f'session{session}/s{subject}/'
               f'sess{session_add_zero}_subj{subject_add_zero}_EEG_MI.mat')
        print(url)
        if os.path.exists(
            os.path.join(
                save_dir,
                f'sess{session_add_zero}_subj{subject_add_zero}_EEG_MI.mat'
            )
        ):
            continue
        i = 1
        while i <= retry:
            try:
                obj = SmartDL(url, save_dir, timeout=10)
                obj.start()
                path = obj.get_dest()
                break
            except Exception as e:
                print(e)
                print(f"session {session} | subject {subject} | url {url}")
