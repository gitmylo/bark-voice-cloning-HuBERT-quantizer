import os
import shutil
import zipfile


def prepare(path):
    path = os.path.abspath(path)
    raw_data_paths = {
        'semantic': os.path.join(path, 'semantic'),
        'wav': os.path.join(path, 'wav')
    }
    prepared_path = os.path.join(path, 'prepared')

    if os.path.isdir(prepared_path):  # Already prepared
        return

    os.mkdir(prepared_path)

    offset = 0

    for zip_file in os.listdir(raw_data_paths['semantic']):
        print(f'Extracting {os.path.basename(zip_file)}')
        offset = extract_files({
            'semantic': os.path.join(raw_data_paths['semantic'], zip_file),
            'wav': os.path.join(raw_data_paths['wav'], zip_file)
        }, prepared_path, offset)



def extract_files(zip_files: dict[str, str], out: str, start_offset: int = 0) -> int:
    new_offset = start_offset
    with zipfile.ZipFile(zip_files['semantic'], 'r') as semantic_zip:
        with zipfile.ZipFile(zip_files['wav'], 'r') as wav_zip:
            for file in semantic_zip.infolist():
                for file2 in wav_zip.infolist():
                    if ''.join(file.filename.split('.')[:-1]).lower() == ''.join(file2.filename.split('.')[:-1]):
                        semantic_zip.extract(file, out)
                        shutil.move(os.path.join(out, file.filename), os.path.join(out, f'{new_offset}_semantic.npy'))
                        wav_zip.extract(file2, out)
                        shutil.move(os.path.join(out, file2.filename), os.path.join(out, f'{new_offset}_wav.wav'))
                        new_offset += 1
    return new_offset

def prepare_2(path):

