import gdown
from enum import StrEnum
import re
from pathlib import Path

import pandas as pd


class Phase(StrEnum):
    SYS = 'sys'
    DIA = 'dia'
    SYS_MASK = 'sys_seg'
    DIA_MASK = 'dia_seg'

# all the patient folders in the train folder
patient_urls = ['1pAfMQ0-NgksPVftTz7XceMdwhya0CFs3', '1yd3qtHzWUzSF3dVGdFtYBG2a9LMaaAon', '12Lh2F-ZNL9dYLsOZioa2_sh-Ep0EvhuR', '1jEh9FlfYpTXgdo_2159tFFZRooxWd-Xg', '1VwuGljN1AsH82roHcEgE0FUbQ598PaYB', '1DLyfNO0VckBbmpoqb-yZM8bAh99Xgk1D', '1WaGyhlLw7jsgB2DBSx3Q7u0lMsgK_386', '1-84HLpPZJwxUD0X3zZkx0tAKfypGpXlH', '1jAvy_8TXM-sMfQvQWf0fJnVWLURNyx3v', '1msq39FXtzNE2fhIqLn03OKrlPkxT4Fd7', '1LIT1alM2w7Y-Vkt5wt2CaTTI2oSLiJtE', '1ZLmNOtbdBPrzXjiLzzN2OuR_eU67LyZl', '12bl89tVPDNhYfTsiea8gG74Rl3DMQx6y', '1wQb00pJ0NjxFqtsmfvW3ZuhGyAFyEK5b', '16aWjTh8xakwi2afruFydZpYolsq5crwm', '1hc6152NfiL_gNnVzbq5tUQ2aTIF8-e5j', '1T2HoMUjpLh1JVWUngFeWmrvv1E2DH538', '12BT20dUxoF9KaPPibb9GqMc7IIrKXAhN', '1t3F_uiWqkMsOoeKgm78zADMU5H-APBl8', '14rnCy_qu8I3ZX2IaoWi218q4esw6aMnr', '10lOxUH5h5ktQv3xLmhAr43cUYGwnkq63', '1NT3b5c2gcsiTufA8rcNeUlLHk0oLDfri', '1TFskgcjcbDjTKcMU77OCgGA2c-0MnGxA', '1Bobrsv4cTMXgD1yCbkYLTv94wybBqY6A', '1zObAIHZU0LVcGx6mLT_U_9xMLuauMQpY', '1Dc3srigda4VdgFwWDSx0tkIGaitHCJNe', '1yucn6roFNbcAh0sIIPo0OES9hSFmOANG', '1wqI182IM5cMNQBzDY0cIXeSIrRrnkp_Q', '1dvrU70r1YxcpMDpNoGsKJeDw0LYBXokM', '1SqkA-CNJP7CmRGwEExOmHFPEr2Awsqqj', '12vVppeMHAlwQONOAaaw-92ns_0RS241z', '1dyuIjkyAaa7k0w8FU2N69yKjZHiJfO3w', '1wl3daUKGU45_5Mb3jiONy6tiXsgKf28i', '15t_Lk8b38SqUoU4Y29e2NKov7DY5qj6m', '15vLXV7jWQSVnHm1y_m0q8qPYW0mhcaEF', '1JMipuzZ80AuqZyJOYknIpijFLmbp9pdi', '1VW2WhOCJNZ7QZ2L6c86t4tBLg9zlNTyI', '1V0Pfwu2QlVMpK-Ts9ZV_pR1tjC1oNAQ6', '1RgQtHGVcW43p1ZebtxPj1PJtsrtFr2BC', '1PPXpXOQErFmcxD1rzDbpw2oEQ3VM2ClB', '1CB8vsiwbxCXPee-JFvv6P13ccDl0YWK3', '1SQ4JWAY7CWZi_v9sUGXbdWB73a8Bm6XH', '1ReEVEQ-_BABXoYmLSBNpGtSuZm_4hxMu', '1ptP4HpZliPHEIY-TKncqVnutaNDFGxoA', '18DKrJQxruYHrI010VAoFDbuFa30vLxtF', '1MCC1iq-v454mo-Pr_jGHphIkn_WWcJaj', '1V1j56DKVIH-aDilpqDimUs6FhhAwyZUB', '19agS_tmWOeuKcpRj9YSlR2utNZ8o6hLj', '1LeR9Y8q0vwUiB1EffuuJS69KWy55A3YH', '1oJsgCn15IZwhkCTkX4WQva7mY1P1o8Qd', '18Kjx7VzQQk2tu_NAyic_Fm5Qs6sck4oj', '1qRGK5RejZ1mTQHVv8P_xTm0lWhGdWGGn', '1nkkzAiHZiUa2cmEiGfLxfjUy8pXLx4Rp', '1M7jdTE-id37_4nllfY9R7iDaMPxayAVC', '19QkcWuoefpTMPXxLx0l_c5TI7qWkQ6Ds', '1xL1birb06fqin0NRCqELQN6acnQQhh0n', '1ak4Bf17tX_CuXMrLKecXMEMoY4SH1fCi', '1s1iNt2MsZaWqbSKTh7nBJO4DPDe9hp8P', '1_VdkwGOxwNd9b4IzscZ2lD0cqyWDiexu', '1c2GPCNpjJkbP0RBFD0Ishp6OatjzLk0-', '1uMlpIZuYM78hC190G-ISGvO2A1Z5X3Ou', '1vsZMnir6oFs8s-tiQnh7a-Pxez27tija', '1Hp0rPJXsi09Nbu0z8BWB6mV1dEVTKxkY', '10gp2CVOkUrjQOtSXm9xD5oV9IoWTlfki', '1fK4ZuETdjJiosfmv6aKIkKljQJKYuPZI', '15L0fYjuDCD3f4d5G4UskrIYdB3nxZw_C', '1pgdSjFL2lKG113YrxuAVeFH7Cyt5OxOt', '1NVJQUdr5yC5pykcQ2QR62hB6OzDUl7xN', '1HtoIeW6OiGvvwPzCr4q1sgMqDPFMR5dF', '1pp7-18wGBc7XEG4mkfUc-nVta5jMo5bU', '1BaM5aqB992k8a8SyvllyHKL9MPEVXQmC', '11wCOuD4AN7gWe7VOd9zIxXp2M1ppCgWz', '1oMQ_xv4eVHvJLLbVjx072r8MJQb8Z_47', '11_mkO8Vv8O0kQfdzjUw2546DZGNcPy8L', '1fCtEpToB3KjMwFgRyZ1UhdT8bxin9BCS', '1DzaFrpoMG8n5ne9M230cD_6PThJ68zH8', '12q7YNB2mzyjp0148xVIX7ve6fnf-6Ki_', '1RqUB5Rp2wjVG2K7OOfqUJwsw4sI40BkC', '1S5JbWscjyESsCrJ-Q0-zWoSyCg12Nhb9', '15K_b85Af50Pj8jRP1UttorwOiTFPL0j0', '13gmpNDpZa1jinfVmLWZqRI9UiKAcNKw-', '1i-sq42JO_7MglCJBPynEPmLlw9OxwB3z', '1S6yYf5HO397rKfoLB4864JbsfopnKd5T', '1VxSgQvf_PFcmNwV78nXbHIf7LP0fwwJc', '1WzI6H5ZuomH6Y2k5CXj2n5NwuBK-Z3Bv', '1h4UjF329plJg2SqakV8wTK7qek6oqmub', '1HeKQbDwUa08jzmjuCjHIqouXtiMys11V', '1MOVPpuhXg1Fzz8dWfFT6pS3adSmVN9CV', '12By9bETwqFh2IZw6NMTzkYz78rNmlv8l', '1nvnpBlbC8l2N7i5xOxW6v9l4Gm_zE1EU', '1xtw8NGR21oEf9s0YymMMJlYJtQwUkypI', '18UQFrrWoX9qXP1lGEQNhDsShZIVlfOAs', '1CQNKicupfCK3A_Hd_gG7LjWUXyGK5Jfe', '1cx4ksTXCkRUXsSJYTOgseTqjIm9zUO-a', '1hlWy2gy3Ooj7TEBe9QKL-3XssXxEDry6', '1TM4F9dGjWvivlcBg3IImBa_-9_aGfWzK', '1ktXKzPVOVoeoVr0AQxUYaP08_n9lBeHw', '17NYKj9gOdq9QDYxHooUw7wRp5G-Rkhmg', '1OmWRAnfJu1zW7AntdlIHD15i10y-0XkR', '1LWTKJISBdorq-PsBDSeri_f8sC0gTjun', '1kRN2mDXhVHF4yNn9YV2oOifuc0xVKY1_']


def download_train_data(out='../data/train'):
    for k in range(100):
        sid = k+1
        download_patient(sid, out)


def download_test_data(out='../data/test'):
    for k in range(100, len(patient_urls)):
        sid = k+1
        download_patient(sid, out)


def download_patient(sid, out='../data/train'):
    folder = Path(out) / f'p{sid:04d}'
    if folder.exists():
        return folder
    folder.mkdir(exist_ok=True, parents=True)
    print(f"Downloading patient {sid} to {folder}...")
    result = gdown.download_folder(id=patient_urls[sid-1], output=str(folder), quiet=True, resume=True)
    if result is None:
        raise Exception(f"Failed to download patient {sid}")
    return folder



# load the content of one patient
def load_patient(sid, parent_dir='../data/train', load_imgs=False):
    patient_folder = download_patient(sid, parent_dir)
    data = get_subj_data(patient_folder, load_imgs=load_imgs)
    return data


# load the content of all patients
def load_all_patients(parent_dir='../data/train', load_imgs=False, return_df=False):
    parent_dir = Path(parent_dir)
    data = [get_subj_data(f, load_imgs=load_imgs) for f in sorted(parent_dir.iterdir())]
    if return_df:
        return pd.DataFrame(data).set_index('id')
    else:
        return data


# returns patient data as structured dict
def get_subj_data(patient_folder, load_imgs = False):
    import nibabel as nib

    name = patient_folder.stem
    data = {'id': name}
    if (patient_folder / 'gt.txt').exists():
        data['label'] = (patient_folder / 'gt.txt').read_text()
    for frame in patient_folder.glob('*.nii.gz'):
        frame_id, gt = re.search(r"p\d+_frame(\d+)(_gt)?", frame.stem).groups()
        phase = Phase.SYS if int(frame_id) < 5 else Phase.DIA
        key = f"{phase}{'' if gt is None else '_seg'}"
        if load_imgs:
            nii_img = nib.load(frame)
            data[key] = nii_img.get_fdata()
            data[f'{key}_header'] = nii_img.header
        else:
            data[key] = frame
    return data

def print_patient_data(subj):
    print(f'Patient {subj["id"] if 'id' in subj else subj.index} ({subj["label"]})')
    print({key.value: subj[key].shape for key in [Phase.SYS, Phase.DIA, Phase.SYS_MASK, Phase.DIA_MASK]})
    print(f'Scaling: {subj["sys_header"].get_zooms()}')