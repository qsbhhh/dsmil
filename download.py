import urllib.request
import argparse
from tqdm import tqdm
import zipfile
import shutil
import os


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
def unzip_data(zip_path, data_path):
    os.makedirs(data_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mil', help='Dataset to be downloaded: mil, tcga')
    args = parser.parse_args()
    
    if args.dataset == "mil":
        print('downloading MIL benchmark datasets')
        download_url('https://uwmadison.box.com/shared/static/arvv7f1k8c2m8e2hugqltxgt9zbbpbh2.zip', '/hdd/xb/mil-dataset.zip')
        unzip_data('/hdd/xb/mil-dataset.zip', '/hdd/xb/datasets')
        os.remove('/hdd/xb/mil-dataset.zip')
    if args.dataset == "tcga":
        print('downloading TCGA Lung Cancer datasets (pre-computed features)')
        download_url('https://uwmadison.box.com/shared/static/tze4yqclajbdzjwxyb8b1umfwk9vcdwq.zip', '/hdd/xb/tcga-dataset.zip')
        unzip_data('/hdd/xb/tcga-dataset.zip', '/hdd/xb/datasets')
        os.remove('/hdd/xb/tcga-dataset.zip')
    if args.dataset == "c16":
        print('downloading Camelyon16 datasets (pre-computed features)')
        download_url('https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip', '/hdd/xb/c16-dataset.zip')
        unzip_data('/hdd/xb/c16-dataset.zip', '/hdd/xb/datasets/Camelyon16')
        os.remove('/hdd/xb/c16-dataset.zip')
    if args.dataset == "tcga-test":
        print('downloading TCGA Lung Cancer testing datasets')
        download_url('https://uwmadison.box.com/shared/static/q4d9fr93wmllf1navjf2ghc9b0pmzf2a.zip', '/hdd/xb/tcga-dataset-test.zip')
        try:
            unzip_data('/hdd/xb/tcga-dataset-test.zip', '/hdd/xb/test/input')
            os.remove('/hdd/xb/cga-dataset-test.zip')
        except:
            print('Please unzip `tcga-dataset-test.zip` to `test/input` manually.')
        download_url('https://uwmadison.box.com/shared/static/grxja488s4i07h9wo3tm4sj6t4nqtz0b.zip', '/hdd/xb/test-weights.zip')
        unzip_data('/hdd/xb/test-weights.zip', '/hdd/xb/test/weights')
        os.remove('/hdd/xb/test-weights.zip')
    if args.dataset == "c16-test":
        print('downloading Camelyon16 testing datasets')
        download_url('https://uwmadison.box.com/shared/static/6t7qcmhv5g1a42qaxmuffpvspqzcw1aq.zip', '/hdd/xb/c16-dataset-test.zip')
        try:
            unzip_data('/hdd/xb/c16-dataset-test.zip', '/hdd/xb/test-c16/input')
            os.remove('/hdd/xb/c16-dataset-test.zip')
        except:
            print('Please unzip `c16-dataset-test.zip` to `test-c16/input` manually.')
        download_url('https://uwmadison.box.com/shared/static/qs717clgaux5hx2mf5qnwmlsoz2elci2.zip', '/hdd/xb/test-weights-c16.zip')
        unzip_data('/hdd/xb/test-weights-c16.zip', '/hdd/xb/test-c16/weights')
        os.remove('/hdd/xb/test-weights-c16.zip')
    
if __name__ == '__main__':
    main()