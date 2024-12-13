from PIL import Image
from fontTools.ttLib.tables.ttProgram import instructions
from qiskit.circuit.library.standard_gates.equivalence_library import instr

from distributedCircuit import distributedAngleSimulation, distributedBasisSimulation
from imageUtil import *
from generalImageRepresentation import *
from circuits import amplitudeEncoding, simulate, angleEncodingCircuit, basisEncoding
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from imgRev import ampDisReversion, generateKeySet, angleDisReversion
from qiskit_backend import constructBackend

IMG_PATH = 'img/duck.png'

def experiments(experSettings):
    vec = image2Arr(imagePath=experSettings['originalImgPath'], size=experSettings['resize'])
    shots = experSettings['shots']
    result_home = experSettings['home_path']
    # create image dict list
    imgDictList = [{
        'img_path': experSettings['originalImgPath'],
        'title': 'Original'
    }]

    # create diff image dict list
    imgDiffList = [{
        'img_path': experSettings['originalImgPath'],
        'title': 'Original'
    }]
    # Amplitude Encoding Experiments
    if experSettings['encoding'] == 'Amplitude Encoding':
        # create circuit
        qc, sqSum, n = amplitudeEncoding(vec)
        keySet = generateKeySet(n)
        # build ideal simulator
        ideal_sim = constructBackend('statevector', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = ampDisReversion(distIdeal, keySet, sqSum, shots, n)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params=param, qb_nums=n,
                                         instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = ampDisReversion(dist, keySet, sqSum, shots, n)
            imgProcessedPath = imageSave(
                img=imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img=pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })
        # print(imgDiffList)

    # Basis Encoding Experiments
    if experSettings['encoding'] == 'Basis Encoding':
        # create circuit
        qc, encodings = basisEncoding(vec)
        n = qc.num_qubits
        # build ideal simulator
        ideal_sim = constructBackend('statevector', 0, n, [])
        imgEncoded = distributedBasisSimulation(vec, 2, shots, ideal_sim)
        # plt.imshow(imgEncoded, cmap='gray')
        # plt.show()
        # distIdeal = simulate(qc, shots, ideal_sim)
        # imgEncoded = angleDisReversion(distIdeal, vec, s, True)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params=param, qb_nums=n,
                                         instructions=experSettings['instructions'])
            imgProcessed = distributedBasisSimulation(vec, 2, shots, noise_sim)
            # dist = simulate(qc, shots, noise_sim)
            # imgProcessed = angleDisReversion(dist, vec, experSettings['shots'], True)
            imgProcessedPath = imageSave(
                img=imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img=pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

    # Angle Encoding Experiments
    if experSettings['encoding'] == 'Angle Encoding':
        # create circuit
        qc = angleEncodingCircuit(vec)
        n = qc.num_qubits
        # build ideal simulator
        ideal_sim = constructBackend('statevector', 0, n, [])
        imgEncoded = distributedAngleSimulation(vec, 8, shots, ideal_sim)
        # plt.imshow(imgEncoded, cmap='gray')
        # plt.show()
        # distIdeal = simulate(qc, shots, ideal_sim)
        # imgEncoded = angleDisReversion(distIdeal, vec, s, True)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
        imgDictList.append({
            'img_path': encodedImgPath,
            'title': 'Encoded Image'
        })

        imgDiffList.append({
            'img_path': encodedImgPath,
            'title': 'Encoded Image'
        })

        #experiment on noise simulator
        for param in tqdm(experSettings['modelParams']):
            noise_sim = constructBackend(method=experSettings['noiseModel'], params = param, qb_nums=n, instructions=experSettings['instructions'])
            imgProcessed = distributedAngleSimulation(vec, 8, shots, noise_sim)
            # dist = simulate(qc, shots, noise_sim)
            # imgProcessed = angleDisReversion(dist, vec, experSettings['shots'], True)
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })
        # print(imgDiffList)

    # MCRQI Experiments
    elif experSettings['encoding'] == 'MCRQI':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='RGB'
        )
        qc, n = MCRQI(img)
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_MCRQI(img, distIdeal, to_print=False)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='RGB')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params = param, qb_nums=n, instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_MCRQI(img, dist, to_print=False)
            # print(type(imgProcessed))
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='RGB'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

        pass

    # FRQI Experiments
    elif experSettings['encoding'] == 'FRQI':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='L'
        )
        qc, n = FRQI(img)
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_FRQI(img, distIdeal)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params = param, qb_nums=n,instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_FRQI(img, dist)
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

    # NEQR Experiments
    elif experSettings['encoding'] == 'NEQR':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='L'
        )
        qc,n = NEQR(img)
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_NEQR(img, distIdeal)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params = param, qb_nums=n, instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_NEQR(img, dist)
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

    # QSMC Experiments
    elif experSettings['encoding'] == 'QSMC':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='L'
        )
        qc = QSMC(img)
        n = qc.num_qubits
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_QSMC(img, distIdeal)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params = param, qb_nums=n, instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_QSMC(img, dist)
            imgProcessedPath = imageSave(
                img = imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img = pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

    # BRQI Experiments
    elif experSettings['encoding'] == 'BRQI':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='L'
        )
        qc = BRQI(img)
        n = qc.num_qubits
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_BRQI(img, distIdeal, qc)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params=param, qb_nums=n,
                                         instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_BRQI(img, dist, qc)
            imgProcessedPath = imageSave(
                img=imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img=pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })

    # OQIM Experiments
    elif experSettings['encoding'] == 'OQIM':
        img = imageOpen(
            imagePath=IMG_PATH,
            size=experSettings['resize'],
            cmap='L'
        )
        qc = OQIM(img)
        n = qc.num_qubits
        ideal_sim = constructBackend('aer', 0, n, [])
        distIdeal = simulate(qc, shots, ideal_sim)
        imgEncoded = Rev_OQIM(img, distIdeal)
        encodedImgPath = imageSave(imgEncoded, 'Encoded', result_home, params=0, imgMode='Gray')
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
            noise_sim = constructBackend(method=experSettings['noiseModel'], params=param, qb_nums=n,
                                         instructions=experSettings['instructions'])
            dist = simulate(qc, shots, noise_sim)
            imgProcessed = Rev_OQIM(img, dist)
            imgProcessedPath = imageSave(
                img=imgProcessed,
                prefix='ampDamp',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
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
            # img_diff = Image.fromarray(pixel_diff)
            # print(img_diff)
            img_diff_path = imageSave(
                img=pixel_diff,
                prefix='Diff',
                resultHome=result_home,
                params=param,
                imgMode='Gray'
            )
            imgDiffList.append({
                'img_path': img_diff_path,
                'title': 'param = ' + str(param),
                'param': param
            })


    # overall diversity
    avgMSE, avgSSIM, weightedDiv = overallPerformanceEval(imgDictList)
    print('The average MSE of this model is {:.3f}.'.format(avgMSE))
    print('The average SSIM of this model is {:.3f}.'.format(avgSSIM))
    print('The Weighted Diversity of this model is {:.3f}.'.format(weightedDiv))

    # plot and save result
    imgPlot(imgDictList, 'out', path=experSettings['home_path'])
    imgPlot(imgDiffList, 'diff', path=experSettings['home_path'])
    plotEvalCurve(imgDictList, path=experSettings['home_path'])

    return imgDictList,imgDiffList

if __name__ == '__main__':

    experiment_settings_AMP = {
        'encoding': 'Amplitude Encoding',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        # 'modelParams': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'shots': 20000,
        'resize': 32,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/AmpEn_AmpDam/',
        # 'instructions': ['measure']
        'instructions': ['u1', 'u2', 'u3']
    }

    experiment_settings_Basis = {
        'encoding': 'Basis Encoding',
        'noiseModel': 'Bit Flip',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        # 'modelParams': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'shots': 1,
        'resize': 32,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/Basis_BitFlip/',
        # 'instructions': ['measure']
        'instructions': ['u1', 'u2', 'u3']
    }

    experiment_settings_Angle = {
        'encoding': 'Angle Encoding',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        # 'modelParams': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'shots': 20000,
        'resize': 32,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/Angle_AmpDam/',
        # 'instructions': ['measure']
        'instructions': ['u1', 'u2', 'u3']
    }

    experiment_settings_MCRQI = {
        'encoding': 'MCRQI',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 32,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/MCRQI_AmpDam/',
        'instructions': ['measure']
        # 'instructions': ['u1', 'u2', 'u3']
    }

    experiment_settings_FRQI = {
        'encoding': 'FRQI',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 16,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/FRQI_AmpDam/',
        # 'instructions': ['u1', 'u2', 'u3']
        'instructions': ['measure']
    }

    experiment_settings_NEQR = {
        'encoding': 'NEQR',
        'noiseModel': 'Depolarization',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 8,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/NEQR_Depolar/',
        'instructions': ['u1', 'u2', 'u3']
    }

    experiment_settings_QSMC = {
        'encoding': 'QSMC',
        'noiseModel': 'Amplitude Damping',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 16,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/QSMC_AmpDam/',
        # 'instructions': ['u1', 'u2', 'u3']
        'instructions': ['measure']
    }

    experiment_settings_OQIM = {
        'encoding': 'OQIM',
        'noiseModel': 'Depolarization',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 16,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/OQIM_Depolar/',
        'instructions': ['u1', 'u2', 'u3']
        # 'instructions': ['measure']
    }

    experiment_settings_BRQI = {
        'encoding': 'BRQI',
        # 'noiseModel': 'Depolarization',
        'noiseModel': 'Bit Flip',
        'modelParams': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        'shots': 20000,
        'resize': 8,
        'originalImgPath': IMG_PATH,
        'home_path': 'result/BRQI_BitFlip/',
        'instructions': ['u1', 'u2', 'u3']
        # 'instructions': ['measure']
    }

    imgDictList,imgDiffList = experiments(experiment_settings_Basis)




