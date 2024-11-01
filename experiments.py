from PIL import Image
from imageUtil import image2Arr, to_bin, imageSave, imgPlot, imageEval, plotEvalCurve
from circuits import amplitudeEncoding, simulate
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from imgRev import ampDisReversion, generateKeySet
from qiskit_backend import constructBackend

IMG_PATH = 'img/I08.png'

def experiments(experSettings):
    vec = image2Arr(imagePath=experSettings['originalImgPath'], size=experSettings['resize'])
    shots = experSettings['shots']
    result_home = experSettings['home_path']
    # create image dict list
    imgDictList = [{
        'img_path': IMG_PATH,
        'title': 'Original'
    }]

    # create diff image dict list
    imgDiffList = [{
        'img_path': IMG_PATH,
        'title': 'Original'
    }]
    if experSettings['encoding'] == 'Amplitude Encoding':
        # create circuit
        qc, sqSum, n = amplitudeEncoding(vec)
        keySet = generateKeySet(n)
        # build ideal simulator
        ideal_sim = constructBackend('statevector', 0, n)
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = ampDisReversion(distIdeal, keySet, sqSum, shots, n)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0)
        imgDictList.append({
            'img_path': encodedImgPath,
            'title': 'Encoded Image'
        })

        imgDiffList.append({
            'img_path': encodedImgPath,
            'title': 'Encoded Image'
        })

        # experiment on noise simulator
        for param in tqdm(experSettings['modelParams']):
            noise_sim = constructBackend(method='Amplitude Damping', params = param, qb_nums=n)
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = ampDisReversion(dist, keySet, sqSum, shots, n)
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param
            )

            mse, ssim = imageEval(encodedImgPath, imgProcessedPath)
            imgDictList.append({
                'img_path': imgProcessedPath,
                'title': 'param = ' + str(param),
                'param': param,
                'mse': mse,
                'ssim': ssim
            })

            # create diff images
            processed_img = cv2.imread(imgProcessedPath)
            encoded_img = cv2.imread(encodedImgPath)
            pixel_diff = cv2.absdiff(processed_img, encoded_img)
            img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = img_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })
        # print(imgDiffList)
    elif experSettings['encoding'] == 'Dense Angle Encoding':
        pass

    return imgDictList,imgDiffList

if __name__ == '__main__':
    experiment_settings = {
        'encoding': 'Amplitude Encoding',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 32,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/'
    }
    imgDictList,imgDiffList = experiments(experiment_settings)
    imgPlot(imgDictList)
    imgPlot(imgDiffList)
    plotEvalCurve(imgDictList)



