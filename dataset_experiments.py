import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from circuits import amplitudeEncoding, simulate
from imageUtil import image2Arr
from imgRev import generateKeySet, ampDisReversion
from qiskit_backend import constructBackend
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


# experiments on datasets
def experiments_dataset(experSettings):
    encoding = experSettings['encoding']
    dataset_path = experSettings['dataset']
    noise_model = experSettings['noiseModel']
    param = experSettings['modelParam']
    shots = experSettings['shots']
    resize = experSettings['resize']
    result_home=experSettings['homePath']
    instructions = experSettings['instructions']

    # generate dataset
    cateList = datasetGenerate(dataset_path, resize)

    # Running experiments
    print('Running experiments...')
    evalDict = {}
    for cate_path in tqdm(cateList, desc='Outer Loop', leave=True):
        # print(cate_path[8:]）
        evalDict[cate_path[8:]] = {
            'avgMSE':0,
            'avgSSIM':0,
            'avgPSNR':0
        }
        mses = []
        ssims = []
        psnrs = []
        for img_name in tqdm(os.listdir(cate_path), desc='Inner Loop', leave=False):
            img_path = os.path.join(cate_path, img_name)
            vec = image2Arr(imagePath=img_path, size='NoResize')
            # create circuit
            qc, sqSum, n = amplitudeEncoding(vec)
            keySet = generateKeySet(n)
            # ideal simulation
            ideal_sim = constructBackend('statevector', 0, n, [])
            distIdeal = simulate(qc, shots, ideal_sim)
            imgEncoded = ampDisReversion(distIdeal, keySet, sqSum, shots, n)
            # noise simulation
            noise_sim = constructBackend(
                method=noise_model,
                params=param,
                qb_nums=n,
                instructions=instructions
            )
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = ampDisReversion(dist, keySet, sqSum, shots, n)
            mses.append(mean_squared_error(imgEncoded, imgProcessed))
            ssims.append(structural_similarity(imgEncoded, imgProcessed, data_range=imgEncoded.max()-imgEncoded.min()))
            psnrs.append(peak_signal_noise_ratio(imgEncoded, imgProcessed, data_range=imgEncoded.max()-imgEncoded.min()))
        evalDict[cate_path[8:]]['avgMSE'] = np.mean(mses)
        evalDict[cate_path[8:]]['avgSSIM'] = np.mean(ssims)
        evalDict[cate_path[8:]]['avgPSNR'] = np.mean(psnrs)
    print('The experiment result is:')
    keys = evalDict.keys()
    for key in keys:
        print('The average MSE of data category {} is {:.3f}'.format(key, evalDict[key]['avgMSE']))
        print('The average SSIM of data category {} is {:.3f}'.format(key, evalDict[key]['avgSSIM']))
        print('The average PSNR of data category {} is {:.3f}'.format(key, evalDict[key]['avgPSNR']))
    print(evalDict)

# return qc
def qcGenerate(encoding, vec):
    if encoding == 'Amplitude Encoding':
        qc, sqSum, n = amplitudeEncoding(vec)
        keySet = generateKeySet(n)
        return qc, sqSum, keySet

# generate dataset
def datasetGenerate(dataset, resize):
    datasetDict={}
    dataset_path = os.listdir(dataset)
    test_path = os.path.join(dataset, dataset_path[0])
    train_path = os.path.join(dataset, dataset_path[1])
    categories = os.listdir(test_path)

    for category in categories:
        datasetDict[category] = []
        img_test_list = os.path.join(test_path, category)
        img_train_list = os.path.join(train_path, category)
        for test_item in os.listdir(img_test_list):
            img_path_test = os.path.join(img_test_list, test_item)
            datasetDict[category].append(img_path_test)
        for train_item in os.listdir(img_train_list):
            img_path_train = os.path.join(img_train_list, train_item)
            datasetDict[category].append(img_path_train)

    print('Generating dataset...')
    cateList = []
    for category in tqdm(categories):
        catePath = os.path.join('dataset', category)
        cateList.append(catePath)
        if not os.path.exists(catePath):
            os.makedirs(catePath)
        for index in range(len(datasetDict[category])):
            img = Image.open(datasetDict[category][index]).convert('L')
            resized_img = img.resize((resize, resize))
            new_path = os.path.join(catePath, str(index)+'.jpg')
            resized_img.save(new_path)

    return cateList

if __name__ == '__main__':
    experiment_settings_AMP_Linnaeus = {
        'encoding': 'Amplitude Encoding',
        'dataset': 'C:\\Users\\Owner\\Downloads\\Linnaeus',
        'noiseModel': 'Amplitude Damping',
        'modelParam': 0.2,
        'shots': 20000,
        'resize': 32,
        'homePath': 'result/datasets/Linnaeus',
        'instructions': ['measure']
    }

    experiments_dataset(experiment_settings_AMP_Linnaeus)
