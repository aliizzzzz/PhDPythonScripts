#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:00:53 2022q

@author: alizz
"""

from screeninfo import get_monitors
from IPython import get_ipython
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from sys import exit, stdout
import os
import pandas as pd
from time import sleep

# screen = get_monitors()[0]
# SCREENW, SCREENH = screen.width, screen.height

VIDEOPATH = '../Experiments/Videos/'
FLOWRATE = '10mlpm/'
FILENAME = 'DSS-1-4-rep3.mp4'
TUBELENGTH = 100  # mm
THRESHOLD = 215


# %% Functions
def q():
    cv2.destroyAllWindows()
    plt.close('all')


def print_message(image, text):
    y0, dy = int(image.shape[0])*2//3, 40
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        if i == 0:
            cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),  3, cv2.LINE_AA)
        else:
            cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),  2, cv2.LINE_AA)
    return image


def get_crop_coordinates(frame):
    ''' Detect the crop coordinates from the image '''
    text = 'Crop the region of interest\n'
    text += '1) Click on the top left corner of the crop rectangle\n'
    text += '2) Click on the bottom right corner\n'
    text += '3) Press any key to continue'
    tempImage = frame.copy()
    tempImage = print_message(tempImage, text)
    pt = []
    num_clicks = 0

    def getxy(event, x, y, flags, param):
        nonlocal pt, num_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(tempImage, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow(FILENAME, tempImage)
            pt.append((x, y))
            num_clicks += 1
            if num_clicks == 2:
                cv2.rectangle(tempImage, pt[0], pt[1], (0, 255, 0), 2)
                cv2.imshow(FILENAME, tempImage)

    cv2.namedWindow(FILENAME)
    cv2.setMouseCallback(FILENAME, getxy)
    cv2.imshow(FILENAME, tempImage)
    _, _, w, h = cv2.getWindowImageRect(FILENAME)
    # cv2.moveWindow(FILENAME, SCREENW//2, 0)
    cv2.setWindowProperty(FILENAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        pass
    cv2.destroyAllWindows()
    return pt


def get_needle_coordinates(frame):
    ''' Detect the needle tip coordinates from the image '''
    text = 'Select needle tip\n'
    text += '1) Click on the tip of the needle\n'
    text += '2) Press any key to continue'
    tempImage = frame.copy()
    tempImage = print_message(tempImage, text)
    num_clicks = 0
    pt = 0

    def getxy(event, x, y, flags, param):
        nonlocal pt, num_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(tempImage, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow(FILENAME, tempImage)
            pt = (x, y)
            num_clicks += 1
            if num_clicks == 1:
                cv2.imshow(FILENAME, tempImage)

    cv2.namedWindow(FILENAME)
    cv2.setMouseCallback(FILENAME, getxy)
    cv2.imshow(FILENAME, tempImage)
    _, _, w, h = cv2.getWindowImageRect(FILENAME)
    # cv2.moveWindow(FILENAME, SCREENW//2, 0)
    cv2.setWindowProperty(FILENAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        pass
    cv2.destroyAllWindows()
    return pt


def get_calibration_line_length(frame):
    ''' Detect the coordinates from the image '''
    # frame = cv2.imread (f 'data / {FILENAME} _0.jpg')
    text = 'Define the calibration line\n'
    text += '1) Click on beginning of the ROI\n'
    text += '2) Click on end of the ROI\n'
    text += '3) Press any key to continue\n'
    tempImage = frame.copy()
    tempImage = print_message(tempImage, text)
    pt = []
    num_clicks = 0

    def getxy(event, x, y, flags, param):
        nonlocal pt, num_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(tempImage, (x, y), 2, (0, 0, 255), 3)
            cv2.imshow(FILENAME, tempImage)
            pt.append((x, y))
            num_clicks += 1
            if num_clicks == 2:
                cv2.line(tempImage, pt[0], pt[1], (0, 255, 0), 2)
                cv2.imshow(FILENAME, tempImage)

    cv2.namedWindow(FILENAME)
    cv2.setMouseCallback(FILENAME, getxy)
    cv2.imshow(FILENAME, tempImage)
    cv2.setWindowProperty(FILENAME, cv2.WND_PROP_TOPMOST, 1)
    _, _, w, h = cv2.getWindowImageRect(FILENAME)
    # cv2.moveWindow(FILENAME, SCREENW//2, 0)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        pass
    cv2.destroyAllWindows()
    x1, y1, x2, y2 = pt[0][0], pt[0][1], pt[1][0], pt[1][1]
    pixel_distance = math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    real_distance = float(TUBELENGTH)

    print()
    print(f'Calibration\n{12*"-"}')
    print(f'{pixel_distance:.2f} pixels')
    print(f'{real_distance:.1f} mm')
    print(f'pixel length: {real_distance/pixel_distance:.5f} mm')
    print(f'pixel area: {(real_distance/pixel_distance)**2:.5f} mm2')

    return pixel_distance, real_distance


def redThresholder(image, thresh):
    ''' red channel is most suitable for segmentation '''
    red = image
    red[:, :, 0] = 0
    red[:, :, 1] = 0

    kernel = np.ones((2, 1), np.uint8)
    it = 7
    _, redthresh = cv2.threshold(red, thresh, 255, cv2.THRESH_BINARY)
    redthresh = cv2.erode(redthresh, kernel, iterations=it)
    redthresh = cv2.dilate(redthresh, kernel, iterations=it)
    return redthresh


def segmentFrame(image, points, thresh):
    ''' Segment Foam '''
    #  red channel is most suitable for segmentation
    redthresh = redThresholder(image, thresh)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(redthresh, cv2.COLOR_BGR2GRAY),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if findFoamContourIndex(contours, points) is None:
        return None
    else:
        foamContourIndex, plugArea = findFoamContourIndex(contours, points)
        justFoam = redthresh.copy()
        justFoam[:, :, :] = 0
        idx = int(foamContourIndex)
        c = contours[idx]
        thickness = -1
        foamColour = (255, 255, 255)
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        top = tuple((bottom[0], top[1]))
        left = tuple((left[0], (right[1]+left[1])//2))
        right = tuple((right[0], left[1]))
        cv2.drawContours(justFoam, contours, idx, foamColour, thickness)
        cv2.line(justFoam, left, right, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(justFoam, top, bottom, (255, 0, 255), 2, cv2.LINE_AA)
        contourExtremes = tuple((top, bottom, left, right))
        return justFoam, plugArea, contourExtremes


def findFoamContourIndex(contours, points):
    contourContainsNeedle = []
    for c in contours:
        if cv2.pointPolygonTest(c, points, False) == 1:
            contourContainsNeedle.append(True)
        else:
            contourContainsNeedle.append(False)
    idx = np.squeeze(np.argwhere(contourContainsNeedle))
    if idx.size == 0:
        return None
    else:
        area = cv2.contourArea(contours[idx])
        return (idx, area)


def updateNeedlePoints(img, needle):
    def moveRight(points):
        newPoints = (points[0]+1, points[1])
        return newPoints

    def moveLeft(points):
        newPoints = (points[0]-1, points[1])
        return newPoints

    def moveUp(points):
        newPoints = (points[0], points[1]-1)
        return newPoints

    def moveDown(points):
        newPoints = (points[0], points[1]+1)
        return newPoints

    finished = False
    for move in [moveDown, moveUp, moveLeft, moveRight]:
        if finished == True:
            break
        points = needle
        while segmentFrame(img, points, THRESHOLD) is None:
            # tempImage = frame[crop_pt[0][1]:crop_pt[1][1], crop_pt[0][0]:crop_pt[1][0]].copy()
            # tempImage = cv2.rectangle(tempImage, (needle_pt[0]-50, needle_pt[1]-25),
            #                           (needle_pt[0]+50, needle_pt[1]+25), (0, 255, 0), 2)
            # tempImage = cv2.line(tempImage, (0, points[1]), (tempImage.shape[1], points[1]),
            #                      (0, 0, 255), 1, cv2.LINE_AA)
            # tempImage = cv2.line(tempImage, (points[0], 0), (points[0], tempImage.shape[0]),
            #                      (0, 0, 255), 1, cv2.LINE_AA)
            # tempImage = cv2.circle(tempImage, points, 1, (0, 255, 0), 2)
            # cv2.imshow('Needle relocation', tempImage)
            # cv2.setWindowProperty('Needle relocation', cv2.WND_PROP_TOPMOST, 1)
            points = move(points)
            # sleep(0.1)
            # print(points)
            if (points[0] >= needle_pt[0]+50 or points[1] >= needle_pt[1]+25 or 
                points[0] < needle_pt[0]-50 or points[1] < needle_pt[1]-25):
                break
            if segmentFrame(img, points, THRESHOLD) is not None:
                print(f'\nFoam found! detection point recalculated to ({points[0]},{points[1]})!\n')
                finished = True
                break
        else:
            continue
    return points


if __name__ == '__main__':
    get_ipython().magic('reset -sf')
    get_ipython().magic('clear')  # clear the console
    plt.close('all')
    # %% Load video
    cap = cv2.VideoCapture(VIDEOPATH + FLOWRATE + FILENAME)  # Open the video
    if not cap.isOpened():  # if there are errors exit
        print('Error opening video stream or file')
        exit(1)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cv2.CAP_PROP_FPS
    print(f'\n\n\nProcessing {FLOWRATE+FILENAME}\n{23*"="}')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # %% Initialise parameters
    foam_detected = False
    originalImages = []
    segmentedImages = []
    time = []
    areas = []
    foamHeight = []
    foamLength = []
    foamExtremePts = []
    frontVelocity = []
    # %% Process frames
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if ret is True:
            if cur_frame == 0:  # Get user inputs at the first frame
                print('Total frames: %i' % totalFrames)
                print('Video duration: %.2f s' % (totalFrames/fps))
                print('FPS: %i\n' % fps)
                crop_pt = get_crop_coordinates(frame)
                calibration_pixels, calibration_mm = get_calibration_line_length(frame)
                needle_pt = get_needle_coordinates(frame)
                needle_pt = (needle_pt[0]-crop_pt[0][0], needle_pt[1]-crop_pt[0][1])
                frameSkip = int(input('Enter skip-rate (Enter 1 to not skip any frames):\n>> '))
                delay = int(input('\nHow many seconds delay before injection starts?\n>> '))
                print()
                # crop_pt = [(28, 872), (1028, 981)]
                # calibration_pixels, calibration_mm = 991.0020181614162, 65
                # needle_pt = (548, 913)
                # needle_pt = (needle_pt[0]-crop_pt[0][0], needle_pt[1]-crop_pt[0][1])
                # frameSkip = 50
                # delay = 50
                # Set starting frame number
                cap.set(cv2.CAP_PROP_POS_FRAMES, fps * delay)
                # Setting pixel calibration variables
                pixelLength = calibration_mm/calibration_pixels  # mm per pixel
                pixelArea = pixelLength**2  # mm2 per pixel
            else:
                # Cut frame
                image = frame.copy()
                image = image[crop_pt[0][1]:crop_pt[1][1], crop_pt[0][0]:crop_pt[1][0]]
                if segmentFrame(image, needle_pt, THRESHOLD) is None and foam_detected is False:
                    stdout.write('\rAnalyzed %i/%i frames, foam not detected' % (cur_frame, totalFrames))
                    stdout.flush()
                    i += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, (fps * delay)+(i * frameSkip))
                    continue
                else:
                    if foam_detected is False:
                        foam_detected = True
                        timeZero = cap.get(cv2.CAP_PROP_POS_MSEC)
                        foamMsg = f'\n\n{10*" "}+{45*"-"}+\n{12*" "}Foam detected '
                        foamMsg += f'at t = {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f} s, '
                        foamMsg += f'frame = {cur_frame}\n{10*" "}+{45*"-"}+\n'
                        print(foamMsg)
                    # segmentedImage, foamArea, extremePts = segmentFrame(image, needle_pt)
                    if cap.get(cv2.CAP_PROP_POS_MSEC) != 0:
                        # Loop to move needle_pt up in case a bubble interferes with foam detection
                        if segmentFrame(image, needle_pt, THRESHOLD) is None:
                            # Show Image
                            # >>>>
                            q()
                            # tempImage = frame[crop_pt[0][1]:crop_pt[1][1], crop_pt[0][0]:crop_pt[1][0]].copy()
                            # tempImage = cv2.rectangle(tempImage, (needle_pt[0]-50, needle_pt[1]-25),
                            #                           (needle_pt[0]+50, needle_pt[1]+25), (0, 255, 0), 2)
                            # tempImage = cv2.line(tempImage, (0, needle_pt[1]), (tempImage.shape[1], needle_pt[1]),
                            #                      (0, 0, 255), 1, cv2.LINE_AA)
                            # tempImage = cv2.line(tempImage, (needle_pt[0], 0), (needle_pt[0], tempImage.shape[0]),
                            #                      (0, 0, 255), 1, cv2.LINE_AA)
                            # tempImage = cv2.circle(tempImage, needle_pt, 1, (0, 255, 0), 2)
                            # cv2.imshow('Needle relocation', tempImage)
                            # cv2.setWindowProperty('Needle relocation', cv2.WND_PROP_TOPMOST, 1)
                            print()
                            print(f'\nframe: {cur_frame}, Bubble detected at ({needle_pt[0]},{needle_pt[1]})')
                            print('Searching needle proximity for foam...')
                            # ui = input('Press "Enter" to search needle proximity for foam, otherwise enter any key:\n>>')
                            # if ui == '':
                            needle_pt = updateNeedlePoints(image, needle_pt)
                            # else:
                                # break
                        segmentedImage, foamArea, extremePts = segmentFrame(image, needle_pt, THRESHOLD)
                        originalImages.append(frame)
                        segmentedImages.append(-1 * segmentedImage + 255)
                        areas.append(foamArea)
                        foamExtremePts.append(extremePts)  # extremePts = (top, bottom, left, right)
                        foamHeight.append(pixelLength*(extremePts[1][1] - extremePts[0][1]))
                        foamLength.append(pixelLength*(extremePts[3][0] - extremePts[2][0]))
                        time.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                        time[-1] = (time[-1]-timeZero)/1000
                        if not frontVelocity:
                            frontVelocity.append(0)
                        else:
                            frontVelocity.append((foamLength[-1]-foamLength[0])/((time[-1]-time[0])/1000))
                        stdout.write('\r%i/%i frames processed!' % (cur_frame, totalFrames))
                        stdout.flush()
                        i += 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (fps * delay)+(i * frameSkip))
                        if round(time[-1], 1) == 5.0:
                            print(f'\n\n{time[-1]:.2f} seconds of foam injecion recorded!\n')
                            break
                    else:
                        stdout.write('\r%i/%i frames processed!' % (cur_frame, totalFrames))
                        stdout.flush()
                        i += 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, (fps * delay)+(i * frameSkip))
        elif foam_detected is False:
            print('\n\nFoam not detected!')
        else:
            print('\n\nVideo analysis completed!   ')
            break
    cap.release()
    # Turning processed daa to Numpy arrays
    originalImages = np.asarray(originalImages, dtype='uint8')
    segmentedImages = np.asarray(segmentedImages, dtype='uint8')
    areas = np.asarray(areas)
    foamHeight = np.asarray(foamHeight)
    foamLength = np.asarray(foamLength)
    frontVelocity = np.asarray(frontVelocity)
    time = np.asarray(time)
    areas = areas * pixelArea  # mm2
    # %% Save images
    plt.style.use('ggplot')
    resultsPath = '../Experiments/Results/'
    experiment = FILENAME[:-4] + '/'
    if os.path.exists(resultsPath) is False:
        os.mkdir(resultsPath)
    if os.path.exists(resultsPath+FLOWRATE) is False:
        os.mkdir(resultsPath+FLOWRATE)
    if os.path.exists(resultsPath+FLOWRATE+experiment) is False:
        os.mkdir(resultsPath+FLOWRATE+experiment)
    for folder in ['original/', 'segmented/', 'surface-area-plot/']:
        if os.path.exists(resultsPath+FLOWRATE+experiment+folder) is False:
            os.mkdir(resultsPath+FLOWRATE+experiment+folder)
    ui = input('\n\nPress "Enter" to save figures, otherwise enter any key: \n>> ')
    # ui = ''
    if ui == '':
        print('\n\nSaving results...\n')
        for i in range(time.size):
            j = i
            stdout.write('\r%.2f %%' % ((j/time.size)*100))
            stdout.flush()
            # Original image
            fig_original, ax = plt.subplots(figsize=(7, 10))
            ax.axis('off')
            img = originalImages[i].copy()
            cv2.putText(img, '%.2f s' % time[i],
                        (img.shape[1]-400, img.shape[0]-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0),  3, cv2.LINE_AA)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.close()
            # Segmented image
            fig_segmented, ax = plt.subplots(figsize=(10, 3.5))
            ax.axis('off')
            bordered = cv2.copyMakeBorder(segmentedImages[i], left=50, right=50, top=100, bottom=100,
                                          borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(bordered, '%.2f s' % time[i],
                        (bordered.shape[1]-300, bordered.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2,
                        cv2.LINE_AA)
            bordered = cv2.copyMakeBorder(bordered, left=2, right=2, top=2, bottom=2,
                                          borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            ax.imshow(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB))
            plt.close()
            # Surface area plot
            fig_surfaceArea, ax = plt.subplots(figsize=(12, 12))
            yax2 = ax.twinx()
            areaPlot = ax.scatter(time[0:i+1], areas[0:i+1], c='k', s=60, marker='+', label='Surface Area')
            heightPlot = yax2.scatter(time[0:i+1], foamHeight[0:i+1], c='g', s=60, marker='+', label='Height')
            lengthPlot = yax2.scatter(time[0:i+1], foamLength[0:i+1], c='r', s=60, marker='+', label='Length')
            ax.grid(True, linestyle='-', linewidth=.5, which='minor')
            ax.grid(True, linestyle='-', linewidth=1.7, which='major')
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlim([0, 5])
            ax.set_ylim(([0, 200]))
            ax.set_xlabel('Time (s)', fontsize=18, labelpad=20)
            ax.set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax.set_title(experiment[:-1], pad=15, fontsize=18, position=[0.5, 1.2], weight='bold')
            yax2.grid(False, which='both')
            yax2.set_ylabel('Distance (mm)', fontsize=18, labelpad=15)
            yax2.tick_params(axis='both', labelsize=14)
            yax2.yaxis.set_minor_locator(plt.MultipleLocator(2))
            yax2.set_ylim((0, 80))
            plots = [areaPlot, heightPlot, lengthPlot]
            labs = [p.get_label() for p in plots]
            ax.legend(plots, labs, fontsize=14, loc='upper right', prop={'family': 'monospace', 'size': 18}, handletextpad=0.8)
            plt.close()
            # Save figures
            fig_original.savefig(f'{resultsPath+FLOWRATE+experiment}original/image{i:03}.png', dpi=150, transparent=False)
            fig_segmented.savefig(f'{resultsPath+FLOWRATE+experiment}segmented/image{i:03}.png', dpi=300, transparent=False)
            fig_surfaceArea.savefig(f'{resultsPath+FLOWRATE+experiment}surface-area-plot/image{i:03}.png', dpi=300,
                                    transparent=False)
            j += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Break loop if 'q' is pressed
                cv2.destroyAllWindows()
                break
            stdout.write('\r%.2f %%' % ((j/time.size)*100))
            stdout.flush()
        print('\nFigures saved!')
    else:
        fig_surfaceArea, ax = plt.subplots(figsize=(12, 12))
        yax2 = ax.twinx()
        areaPlot = ax.scatter(time[0:i+1], areas[0:i+1], c='k', s=60, marker='+', label='Surface Area')
        heightPlot = yax2.scatter(time[0:i+1], foamHeight[0:i+1], c='g', s=60, marker='+', label='Height')
        lengthPlot = yax2.scatter(time[0:i+1], foamLength[0:i+1], c='r', s=60, marker='+', label='Length')
        ax.grid(True, linestyle='-', linewidth=.5, which='minor')
        ax.grid(True, linestyle='-', linewidth=1.7, which='major')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim([0, 5])
        ax.set_ylim(([0, 200]))
        ax.set_xlabel('Time (s)', fontsize=18, labelpad=20)
        ax.set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
        ax.set_title(experiment[:-1], pad=15, fontsize=18, position=[0.5, 1.2], weight='bold')
        yax2.grid(False, which='both')
        yax2.set_ylabel('Distance (mm)', fontsize=18, labelpad=15)
        yax2.tick_params(axis='both', labelsize=14)
        yax2.yaxis.set_minor_locator(plt.MultipleLocator(2))
        yax2.set_ylim((0, 80))
        plots = [areaPlot, heightPlot, lengthPlot]
        labs = [p.get_label() for p in plots]
        ax.legend(plots, labs, fontsize=14, loc='upper right', prop={'family': 'monospace', 'size': 18}, handletextpad=0.8)
    # %% Save animations
    ''' run the following in Terminal @ the FLOWRATE directory:
        for FLOW in *pm; do for EXP in $FLOW/*; do for FOLD in $EXP/*; do ffmpeg -y -framerate 15 -i $FOLD/image%03d.png $FOLD/output.mp4; done; done; done
     '''
    # %% Save surface area data
    ui = input('\nPress "Enter" to save data in .csv files, otherwise enter any key:\n>>')
    # ui = ''
    if ui == '':
        if os.path.exists(resultsPath+'csvs/'+FLOWRATE) is False:
            os.mkdir(resultsPath+'csvs/'+FLOWRATE)
        resultFile = FILENAME[:-4] + '.csv'
        data = pd.DataFrame(np.asarray((time, areas, foamHeight, foamLength, frontVelocity)))
        data = data.transpose()
        data.columns = ['Time (s)', 'Area (mm2)', 'FoamHeight (mm)', 'FoamLength (mm)', 'Front Velocity (mm/s)']
        data.to_csv(resultsPath+'csvs/'+FLOWRATE+resultFile, float_format='%.6f', header=True)
        print(f'\nData saved as {resultsPath}csvs/{resultFile}')

        fig_surfaceArea, ax = plt.subplots(figsize=(12, 12))
        yax2 = ax.twinx()
        areaPlot = ax.scatter(time[0:i+1], areas[0:i+1], c='k', s=60, marker='+', label='Surface Area')
        heightPlot = yax2.scatter(time[0:i+1], foamHeight[0:i+1], c='g', s=60, marker='+', label='Height')
        lengthPlot = yax2.scatter(time[0:i+1], foamLength[0:i+1], c='r', s=60, marker='+', label='Length')
        ax.grid(True, linestyle='-', linewidth=.5, which='minor')
        ax.grid(True, linestyle='-', linewidth=1.7, which='major')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim([0, 5])
        ax.set_ylim(([0, 200]))
        ax.set_xlabel('Time (s)', fontsize=18, labelpad=20)
        ax.set_ylabel(r'Surface area $\mathregular{(mm^{2})}$', fontsize=18, labelpad=15)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
        ax.set_title(experiment[:-1], pad=15, fontsize=18, position=[0.5, 1.2], weight='bold')
        yax2.grid(False, which='both')
        yax2.set_ylabel('Distance (mm)', fontsize=18, labelpad=15)
        yax2.tick_params(axis='both', labelsize=14)
        yax2.yaxis.set_minor_locator(plt.MultipleLocator(2))
        yax2.set_ylim((0, 80))
        plots = [areaPlot, heightPlot, lengthPlot]
        labs = [p.get_label() for p in plots]
        ax.legend(plots, labs, fontsize=14, loc='upper right', prop={'family': 'monospace', 'size': 18}, handletextpad=0.8)
        fig_vel, ax = plt.subplots(figsize=(12, 12))
        ax.plot(data['Time (s)'], data['Front Velocity (mm/s)'])
