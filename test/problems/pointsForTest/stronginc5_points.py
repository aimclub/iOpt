import numpy as np

test_points = np.array([[0.0029296875, 2.997070313, -2.997070313, -9.990234375, -0.009765625, 0, 9.997070313],
                        [-2.997070313, 0.0029296875, 2.997070313, -9.990234375, -0.009765625, 0, 9.997070313],
                        [2.997070313, -0.0029296875, 2.997070313, -9.990234375, -0.009765625, 0, 4.008789063],
                        [2.997070313, -2.997070313, -0.0029296875, 9.990234375, -0.009765625, 0, -9.977539063],
                        [2.997070313, -2.997070313, -0.0029296875, 9.990234375, -0.009765625, 1, 0.5960956573],
                        [2.997070313, -2.997070313, -2.997070313, -0.009765625, 0.009765625, 0, 2.997070313],
                        [2.997070313, -2.997070313, 2.997070313, 0.009765625, 0.009765625, 0, -3.016601563],
                        [2.997070313, -2.997070313, 2.997070313, 0.009765625, 0.009765625, 1, -0.4019512177],
                        [2.997070313, -2.997070313, 2.997070313, 0.009765625, 0.009765625, 2, -40.06262112],
                        [2.997070313, -2.997070313, 2.997070313, 0.009765625, 0.009765625, 3, -11.05776493],
                        [2.997070313, -2.997070313, 2.997070313, 0.009765625, 0.009765625, 4, 16.72487885],
                        [2.997070313, -2.997070313, 2.997070313, 9.990234375, -0.009765625, 0, -12.97753906],
                        [2.997070313, -2.997070313, 2.997070313, 9.990234375, -0.009765625, 1, 0.5960956573],
                        [2.997070313, -2.997070313, 2.997070313, -9.990234375, 0.009765625, 0, 6.983398438],
                        [2.997070313, -0.0029296875, 2.997070313, -4.990234375, 4.990234375, 0, -5.991210938],
                        [2.997070313, -0.0029296875, 2.997070313, -4.990234375, 4.990234375, 1, -1.150974655],
                        [2.997070313, -0.0029296875, 2.997070313, -4.990234375, 4.990234375, 2, -117.763793],
                        [2.997070313, -0.0029296875, 2.997070313, -4.990234375, 4.990234375, 3, 19.03159117],
                        [2.997070313, -0.0029296875, 2.997070313, 5.009765625, 4.990234375, 0, -15.99121094],
                        [2.997070313, -0.0029296875, 2.997070313, 5.009765625, 4.990234375, 1, -1.14902153],
                        [2.997070313, -0.0029296875, 2.997070313, 5.009765625, 4.990234375, 2, -117.763793],
                        [2.997070313, -0.0029296875, 2.997070313, 5.009765625, 4.990234375, 3, 6.367129168],
                        [2.997070313, -2.997070313, -2.997070313, -9.990234375, -0.009765625, 0, 12.99707031],
                        [-2.997070313, -2.997070313, 0.0029296875, 9.990234375, -0.009765625, 0, -3.989257813],
                        [-2.997070313, -2.997070313, 0.0029296875, 9.990234375, -0.009765625, 1, 0.5960956573],
                        [2.997070313, -1.497070313, 0.0029296875, 0.009765625, 4.990234375, 0, -6.502929688],
                        [2.997070313, -1.497070313, 0.0029296875, 0.009765625, 4.990234375, 1, -1.150974655],
                        [2.997070313, -1.497070313, 0.0029296875, 0.009765625, 4.990234375, 2, -117.0225821],
                        [2.997070313, -1.497070313, 0.0029296875, 0.009765625, 4.990234375, 3, 1.361487488],
                        [2.997070313, -1.502929688, 0.0029296875, -0.009765625, 4.990234375, 0, -6.477539063],
                        [2.997070313, -1.502929688, 0.0029296875, -0.009765625, 4.990234375, 1, -1.14902153],
                        [2.997070313, -1.502929688, 0.0029296875, -0.009765625, 4.990234375, 2, -117.0167227],
                        [2.997070313, -1.502929688, 0.0029296875, -0.009765625, 4.990234375, 3, 1.30574689],
                        [0.0029296875, -2.997070313, 1.502929688, -0.009765625, 4.990234375, 0, -3.489257813],
                        [0.0029296875, -2.997070313, 1.502929688, -0.009765625, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -2.997070313, 1.502929688, -0.009765625, 4.990234375, 2, -99.05187893],
                        [0.0029296875, -2.997070313, 1.502929688, -0.009765625, 4.990234375, 3, -14.56714463],
                        [0.0029296875, -2.997070313, 1.502929688, -0.009765625, 4.990234375, 4, 19.18112665],
                        [0.0029296875, -2.997070313, 1.497070313, 0.009765625, 4.990234375, 0, -3.502929688],
                        [0.0029296875, -2.997070313, 1.497070313, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -2.997070313, 1.497070313, 0.009765625, 4.990234375, 2, -99.0577383],
                        [0.0029296875, -2.997070313, 1.497070313, 0.009765625, 4.990234375, 3, -14.59555489],
                        [0.0029296875, -2.997070313, 1.497070313, 0.009765625, 4.990234375, 4, 19.41645072],
                        [-2.997070313, -2.997070313, -2.997070313, 0.009765625, 0.009765625, 0, 8.971679688],
                        [-2.997070313, -2.997070313, 2.997070313, -0.009765625, 0.009765625, 0, 2.997070313],
                        [2.997070313, -1.502929688, 0.0029296875, 9.990234375, 4.990234375, 0, -16.47753906],
                        [2.997070313, -1.502929688, 0.0029296875, 9.990234375, 4.990234375, 1, -0.1509746552],
                        [2.997070313, -1.502929688, 0.0029296875, 9.990234375, 4.990234375, 2, -117.0167227],
                        [2.997070313, -1.502929688, 0.0029296875, 9.990234375, 4.990234375, 3, -0.1622428446],
                        [2.997070313, -1.502929688, 0.0029296875, 9.990234375, 4.990234375, 4, 18.33036969],
                        [-2.997070313, -2.997070313, -2.997070313, -9.990234375, 0.009765625, 0, 18.97167969],
                        [2.997070313, -2.997070313, -2.997070313, 9.990234375, 0.009765625, 0, -7.002929688],
                        [2.997070313, -2.997070313, -2.997070313, 9.990234375, 0.009765625, 1, 0.5960956573],
                        [-2.997070313, -2.997070313, -2.997070313, 9.990234375, -0.009765625, 0, -0.9892578125],
                        [-2.997070313, -2.997070313, -2.997070313, 9.990234375, -0.009765625, 1, 0.5960956573],
                        [-2.997070313, -2.997070313, 2.997070313, 9.990234375, 0.009765625, 0, -7.002929688],
                        [-2.997070313, -2.997070313, 2.997070313, 9.990234375, 0.009765625, 1, 0.5960956573],
                        [2.997070313, -0.0029296875, -2.997070313, 4.990234375, 4.990234375, 0, -9.977539063],
                        [2.997070313, -0.0029296875, -2.997070313, 4.990234375, 4.990234375, 1, -1.150974655],
                        [2.997070313, -0.0029296875, -2.997070313, 4.990234375, 4.990234375, 2, -141.7403555],
                        [2.997070313, -0.0029296875, -2.997070313, 4.990234375, 4.990234375, 3, 0.4133736548],
                        [2.997070313, -1.502929688, -0.0029296875, 0.009765625, 4.990234375, 0, -6.491210938],
                        [2.997070313, -1.502929688, -0.0029296875, 0.009765625, 4.990234375, 1, -1.14902153],
                        [2.997070313, -1.502929688, -0.0029296875, 0.009765625, 4.990234375, 2, -117.0401602],
                        [2.997070313, -1.502929688, -0.0029296875, 0.009765625, 4.990234375, 3, 1.349761031],
                        [2.997070313, -1.497070313, -0.0029296875, 9.990234375, 4.990234375, 0, -16.47753906],
                        [2.997070313, -1.497070313, -0.0029296875, 9.990234375, 4.990234375, 1, -0.1529277802],
                        [2.997070313, -1.497070313, -0.0029296875, 9.990234375, 4.990234375, 2, -117.0460196],
                        [2.997070313, -1.497070313, -0.0029296875, 9.990234375, 4.990234375, 3, -0.1624021412],
                        [2.997070313, -1.497070313, -0.0029296875, 9.990234375, 4.990234375, 4, 18.2330025],
                        [-2.997070313, -0.0029296875, -2.997070313, 5.009765625, 4.990234375, 0, -4.002929688],
                        [-2.997070313, -0.0029296875, -2.997070313, 5.009765625, 4.990234375, 1, -1.14902153],
                        [-2.997070313, -0.0029296875, -2.997070313, 5.009765625, 4.990234375, 2, -129.7520742],
                        [-2.997070313, -0.0029296875, -2.997070313, 5.009765625, 4.990234375, 3, -9.96306455],
                        [-2.997070313, -0.0029296875, -2.997070313, 5.009765625, 4.990234375, 4, 23.66824004],
                        [-2.997070313, -0.0029296875, 2.997070313, 4.990234375, 4.990234375, 0, -9.977539063],
                        [-2.997070313, -0.0029296875, 2.997070313, 4.990234375, 4.990234375, 1, -1.150974655],
                        [-2.997070313, -0.0029296875, 2.997070313, 4.990234375, 4.990234375, 2, -105.7755117],
                        [-2.997070313, -0.0029296875, 2.997070313, 4.990234375, 4.990234375, 3, -3.441990524],
                        [-2.997070313, -0.0029296875, 2.997070313, 4.990234375, 4.990234375, 4, 20.01968147],
                        [0.0029296875, -2.997070313, -1.502929688, 0.009765625, 4.990234375, 0, -0.5029296875],
                        [0.0029296875, -2.997070313, -1.502929688, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -2.997070313, -1.502929688, 0.009765625, 4.990234375, 2, -111.0753164],
                        [0.0029296875, -2.997070313, -1.502929688, 0.009765625, 4.990234375, 3, -11.50419075],
                        [0.0029296875, -2.997070313, -1.502929688, 0.009765625, 4.990234375, 4, 17.23500969],
                        [1.497070313, -2.997070313, -0.0029296875, 0.009765625, 5.009765625, 0, -3.516601563],
                        [1.497070313, -2.997070313, -0.0029296875, 0.009765625, 5.009765625, 1, -0.4019512177],
                        [1.497070313, -2.997070313, -0.0029296875, 0.009765625, 5.009765625, 2, -108.4366446],
                        [1.497070313, -2.997070313, -0.0029296875, 0.009765625, 5.009765625, 3, -3.935213871],
                        [1.497070313, -2.997070313, -0.0029296875, 0.009765625, 5.009765625, 4, 32.61975865],
                        [-2.997070313, -1.502929688, 0.0029296875, 0.009765625, 4.990234375, 0, -0.5029296875],
                        [-2.997070313, -1.502929688, 0.0029296875, 0.009765625, 4.990234375, 1, -1.14902153],
                        [-2.997070313, -1.502929688, 0.0029296875, 0.009765625, 4.990234375, 2, -105.0284414],
                        [-2.997070313, -1.502929688, 0.0029296875, 0.009765625, 4.990234375, 3, -8.999419116],
                        [-2.997070313, -1.502929688, 0.0029296875, 0.009765625, 4.990234375, 4, 22.16447119],
                        [-2.997070313, -1.497070313, 0.0029296875, 9.990234375, 4.990234375, 0, -10.48925781],
                        [-2.997070313, -1.497070313, 0.0029296875, 9.990234375, 4.990234375, 1, -0.1529277802],
                        [-2.997070313, -1.497070313, 0.0029296875, 9.990234375, 4.990234375, 2, -105.0343008],
                        [-2.997070313, -1.497070313, 0.0029296875, 9.990234375, 4.990234375, 3, -10.51158251],
                        [-2.997070313, -1.497070313, 0.0029296875, 9.990234375, 4.990234375, 4, 24.13162932],
                        [0.0029296875, -0.0029296875, -1.502929688, 9.990234375, 4.990234375, 0, -13.47753906],
                        [0.0029296875, -0.0029296875, -1.502929688, 9.990234375, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -0.0029296875, -1.502929688, 9.990234375, 4.990234375, 2, -114.0694571],
                        [0.0029296875, -0.0029296875, -1.502929688, 9.990234375, 4.990234375, 3, -6.062085576],
                        [0.0029296875, -0.0029296875, -1.502929688, 9.990234375, 4.990234375, 4, 2.907443611],
                        [1.502929688, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 0, -13.50292969],
                        [1.502929688, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 1, -0.4019512177],
                        [1.502929688, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 2, -132.4190664],
                        [1.502929688, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 3, 18.86049684],
                        [0.0029296875, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 0, -12.00292969],
                        [0.0029296875, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 1, -0.4019512177],
                        [0.0029296875, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 2, -127.1602774],
                        [0.0029296875, -0.0029296875, -2.997070313, 9.990234375, 5.009765625, 3, 9.846105874],
                        [0.0029296875, -1.497070313, -2.252929688, 9.990234375, 2.509765625, 0, -8.752929688],
                        [0.0029296875, -1.497070313, -2.252929688, 9.990234375, 2.509765625, 1, -0.1529277802],
                        [0.0029296875, -1.497070313, -2.252929688, 9.990234375, 2.509765625, 2, -72.7427969],
                        [0.0029296875, -1.497070313, -2.252929688, 9.990234375, 2.509765625, 3, -4.200688207],
                        [0.0029296875, -1.497070313, -2.252929688, 9.990234375, 2.509765625, 4, 14.87359072],
                        [0.0029296875, -2.997070313, -1.497070313, 9.990234375, 4.990234375, 0, -10.48925781],
                        [0.0029296875, -2.997070313, -1.497070313, 9.990234375, 4.990234375, 1, 0.5960956573],
                        [0.0029296875, -2.997070313, 1.502929688, 9.990234375, 4.990234375, 0, -13.48925781],
                        [0.0029296875, -2.997070313, 1.502929688, 9.990234375, 4.990234375, 1, 0.5960956573],
                        [0.0029296875, -0.0029296875, 1.497070313, 9.990234375, 4.990234375, 0, -16.47753906],
                        [0.0029296875, -0.0029296875, 1.497070313, 9.990234375, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -0.0029296875, 1.497070313, 9.990234375, 4.990234375, 2, -102.0518789],
                        [0.0029296875, -0.0029296875, 1.497070313, 9.990234375, 4.990234375, 3, -3.464557451],
                        [0.0029296875, -0.0029296875, 1.497070313, 9.990234375, 4.990234375, 4, 5.088884635],
                        [1.497070313, -0.0029296875, 2.997070313, 9.990234375, 5.009765625, 0, -19.49121094],
                        [1.497070313, -0.0029296875, 2.997070313, 9.990234375, 5.009765625, 1, -0.4019512177],
                        [1.497070313, -0.0029296875, 2.997070313, 9.990234375, 5.009765625, 2, -108.4132071],
                        [1.497070313, -0.0029296875, 2.997070313, 9.990234375, 5.009765625, 3, 22.51829046],
                        [0.0029296875, -0.0029296875, 2.997070313, 9.990234375, 4.990234375, 0, -17.97753906],
                        [0.0029296875, -0.0029296875, 2.997070313, 9.990234375, 4.990234375, 1, -0.4019512177],
                        [0.0029296875, -0.0029296875, 2.997070313, 9.990234375, 4.990234375, 2, -102.7930899],
                        [0.0029296875, -0.0029296875, 2.997070313, 9.990234375, 4.990234375, 3, 13.60278309],
                        [2.997070313, -1.497070313, 0.0029296875, -9.990234375, 4.990234375, 0, 3.497070313],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 0, -13.25292969],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 1, -0.1529277802],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 2, -54.71642971],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 3, -13.54767125],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 4, -0.237124835],
                        [0.0029296875, -1.497070313, 2.247070313, 9.990234375, 2.509765625, 5, -18.68470052],
                        [0.0029296875, -0.0029296875, 1.502929688, -9.990234375, 4.990234375, 0, 3.497070313],
                        [1.497070313, -0.7470703125, 1.502929688, 9.990234375, 2.509765625, 0, -14.75292969],
                        [1.497070313, -0.7470703125, 1.502929688, 9.990234375, 2.509765625, 1, -0.3399394989],
                        [1.497070313, -0.7470703125, 1.502929688, 9.990234375, 2.509765625, 2, -61.44885159],
                        [1.497070313, -0.7470703125, 1.502929688, 9.990234375, 2.509765625, 3, 2.701199212],
                        [1.497070313, -0.7529296875, 2.997070313, 9.990234375, 2.509765625, 0, -16.24121094],
                        [1.497070313, -0.7529296875, 2.997070313, 9.990234375, 2.509765625, 1, -0.3389629364],
                        [1.497070313, -0.7529296875, 2.997070313, 9.990234375, 2.509765625, 2, -62.18127346],
                        [1.497070313, -0.7529296875, 2.997070313, 9.990234375, 2.509765625, 3, 6.789823337],
                        [0.7470703125, -1.497070313, 2.997070313, 5.009765625, 2.509765625, 0, -9.766601563],
                        [0.7470703125, -1.497070313, 2.997070313, 5.009765625, 2.509765625, 1, -0.8999980927],
                        [0.7470703125, -1.497070313, 2.997070313, 5.009765625, 2.509765625, 2, -57.6959219],
                        [0.7470703125, -1.497070313, 2.997070313, 5.009765625, 2.509765625, 3, -14.57570379],
                        [0.7470703125, -1.497070313, 2.997070313, 5.009765625, 2.509765625, 4, 23.58988008],
                        [-2.997070313, -1.502929688, -0.0029296875, 9.990234375, 4.990234375, 0, -10.47753906],
                        [-2.997070313, -1.502929688, -0.0029296875, 9.990234375, 4.990234375, 1, -0.1509746552],
                        [-2.997070313, -1.502929688, -0.0029296875, 9.990234375, 4.990234375, 2, -105.0518789],
                        [-2.997070313, -1.502929688, -0.0029296875, 9.990234375, 4.990234375, 3, -10.51142314],
                        [-2.997070313, -1.502929688, -0.0029296875, 9.990234375, 4.990234375, 4, 24.0835152],
                        [-2.997070313, -1.497070313, -0.0029296875, 0.009765625, 4.990234375, 0, -0.5029296875],
                        [-2.997070313, -1.497070313, -0.0029296875, 0.009765625, 4.990234375, 1, -1.150974655],
                        [-2.997070313, -1.497070313, -0.0029296875, 0.009765625, 4.990234375, 2, -105.0577383],
                        [-2.997070313, -1.497070313, -0.0029296875, 0.009765625, 4.990234375, 3, -8.987693022],
                        [-2.997070313, -1.497070313, -0.0029296875, 0.009765625, 4.990234375, 4, 22.09082342],
                        [-0.0029296875, -2.997070313, -1.497070313, 0.009765625, 4.990234375, 0, -0.5029296875],
                        [-0.0029296875, -2.997070313, -1.497070313, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [-0.0029296875, -2.997070313, -1.497070313, 0.009765625, 4.990234375, 2, -111.0225821],
                        [-0.0029296875, -2.997070313, -1.497070313, 0.009765625, 4.990234375, 3, -11.5134301],
                        [-0.0029296875, -2.997070313, -1.497070313, 0.009765625, 4.990234375, 4, 17.19297691],
                        [-0.0029296875, -2.997070313, 1.502929688, 0.009765625, 4.990234375, 0, -3.502929688],
                        [-0.0029296875, -2.997070313, 1.502929688, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [-0.0029296875, -2.997070313, 1.502929688, 0.009765625, 4.990234375, 2, -99.04016018],
                        [-0.0029296875, -2.997070313, 1.502929688, 0.009765625, 4.990234375, 3, -14.58496073],
                        [-0.0029296875, -2.997070313, 1.502929688, 0.009765625, 4.990234375, 4, 19.37441793],
                        [-0.0029296875, -2.997070313, -1.502929688, 9.990234375, 4.990234375, 0, -10.47753906],
                        [-0.0029296875, -2.997070313, -1.502929688, 9.990234375, 4.990234375, 1, 0.5960956573],
                        [-0.0029296875, -2.997070313, 1.497070313, 9.990234375, 4.990234375, 0, -13.47753906],
                        [-0.0029296875, -2.997070313, 1.497070313, 9.990234375, 4.990234375, 1, 0.5960956573],
                        [-1.497070313, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 0, 2.471679688],
                        [1.502929688, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 0, -0.5283203125],
                        [1.502929688, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 1, -0.4019512177],
                        [1.502929688, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 2, -129.4249258],
                        [1.502929688, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 3, 9.315337504],
                        [2.997070313, -2.997070313, -2.997070313, 0.009765625, 4.990234375, 0, -2.002929688],
                        [2.997070313, -2.997070313, -2.997070313, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [2.997070313, -2.997070313, -2.997070313, 0.009765625, 4.990234375, 2, -138.7462149],
                        [2.997070313, -2.997070313, -2.997070313, 0.009765625, 4.990234375, 3, 5.549010492],
                        [1.502929688, -2.997070313, -0.0029296875, 9.990234375, 5.009765625, 0, -13.50292969],
                        [1.502929688, -2.997070313, -0.0029296875, 9.990234375, 5.009765625, 1, 0.5960956573],
                        [1.497070313, -2.997070313, 0.0029296875, 9.990234375, 5.009765625, 0, -13.50292969],
                        [1.497070313, -2.997070313, 0.0029296875, 9.990234375, 5.009765625, 1, 0.5960956573],
                        [1.497070313, -2.997070313, 0.0029296875, -0.009765625, 5.009765625, 0, -3.502929688],
                        [1.497070313, -2.997070313, 0.0029296875, -0.009765625, 5.009765625, 1, -0.4019512177],
                        [1.497070313, -2.997070313, 0.0029296875, -0.009765625, 5.009765625, 2, -108.4132071],
                        [1.497070313, -2.997070313, 0.0029296875, -0.009765625, 5.009765625, 3, -3.960479282],
                        [1.497070313, -2.997070313, 0.0029296875, -0.009765625, 5.009765625, 4, 32.39002991],
                        [-1.502929688, -2.997070313, 2.997070313, 0.009765625, 5.009765625, 0, -3.516601563],
                        [-1.502929688, -2.997070313, 2.997070313, 0.009765625, 5.009765625, 1, -0.4019512177],
                        [-1.502929688, -2.997070313, 2.997070313, 0.009765625, 5.009765625, 2, -99.43664455],
                        [-1.502929688, -2.997070313, 2.997070313, 0.009765625, 5.009765625, 3, -17.57582894],
                        [-1.502929688, -2.997070313, 2.997070313, 0.009765625, 5.009765625, 4, 12.69057888],
                        [1.502929688, -2.997070313, 0.0029296875, 0.009765625, 5.009765625, 0, -3.528320313],
                        [1.502929688, -2.997070313, 0.0029296875, 0.009765625, 5.009765625, 1, -0.4019512177],
                        [1.502929688, -2.997070313, 0.0029296875, 0.009765625, 5.009765625, 2, -108.4425039],
                        [1.502929688, -2.997070313, 0.0029296875, 0.009765625, 5.009765625, 3, -3.861347333],
                        [1.502929688, -2.997070313, 0.0029296875, 0.009765625, 5.009765625, 4, 32.48640789],
                        [2.252929688, -1.502929688, -2.997070313, 0.009765625, 2.490234375, 0, -0.2529296875],
                        [2.252929688, -1.502929688, -2.997070313, 0.009765625, 2.490234375, 1, -1.14902153],
                        [2.252929688, -1.502929688, -2.997070313, 0.009765625, 2.490234375, 2, -88.90295315],
                        [2.252929688, -1.502929688, -2.997070313, 0.009765625, 2.490234375, 3, 12.6316833],
                        [-2.997070313, -2.997070313, 2.997070313, 0.009765625, 4.990234375, 0, -2.002929688],
                        [-2.997070313, -2.997070313, 2.997070313, 0.009765625, 4.990234375, 1, -0.4019512177],
                        [-2.997070313, -2.997070313, 2.997070313, 0.009765625, 4.990234375, 2, -102.7813711],
                        [-2.997070313, -2.997070313, 2.997070313, 0.009765625, 4.990234375, 3, -13.71101081],
                        [-2.997070313, -2.997070313, 2.997070313, 0.009765625, 4.990234375, 4, 20.93615148],
                        [1.502929688, -0.0029296875, 2.997070313, -9.990234375, 5.009765625, 0, 0.4833984375],
                        [0.7529296875, -1.497070313, 1.502929688, 5.009765625, 2.509765625, 0, -8.278320313],
                        [0.7529296875, -1.497070313, 1.502929688, 5.009765625, 2.509765625, 1, -0.8999980927],
                        [0.7529296875, -1.497070313, 1.502929688, 5.009765625, 2.509765625, 2, -56.9693594],
                        [0.7529296875, -1.497070313, 1.502929688, 5.009765625, 2.509765625, 3, -4.529679361],
                        [0.7529296875, -1.497070313, 1.502929688, 5.009765625, 2.509765625, 4, 6.827216106],
                        [-1.497070313, -2.997070313, -0.0029296875, 9.990234375, 5.009765625, 0, -10.50292969],
                        [-1.497070313, -2.997070313, -0.0029296875, 9.990234375, 5.009765625, 1, 0.5960956573],
                        [-1.502929688, -2.997070313, 0.0029296875, 9.990234375, 5.009765625, 0, -10.50292969],
                        [-1.502929688, -2.997070313, 0.0029296875, 9.990234375, 5.009765625, 1, 0.5960956573],
                        [-0.0029296875, -2.997070313, -2.997070313, 0.009765625, 4.990234375, 0, 0.9970703125],
                        [0.0029296875, -0.0205078125, -1.543945313, 0.146484375, 4.951171875, 0, -3.536132813],
                        [0.0029296875, -0.0205078125, -1.543945313, 0.146484375, 4.951171875, 1, -1.399738693],
                        [0.0029296875, -0.0205078125, -1.543945313, 0.146484375, 4.951171875, 2, -113.5096273],
                        [0.0029296875, -0.0205078125, -1.543945313, 0.146484375, 4.951171875, 3, -3.185748858],
                        [0.0029296875, -0.0205078125, -1.543945313, 0.146484375, 4.951171875, 4, 3.508961141],
                        [1.456054688, -0.0146484375, -0.0205078125, 0.087890625, 5.029296875, 0, -6.538085938],
                        [1.456054688, -0.0146484375, -0.0205078125, 0.087890625, 5.029296875, 1, -1.399898911],
                        [1.456054688, -0.0146484375, -0.0205078125, 0.087890625, 5.029296875, 2, -111.6430731],
                        [1.456054688, -0.0146484375, -0.0205078125, 0.087890625, 5.029296875, 3, 4.424280486],
                        [1.543945313, -0.0146484375, -2.979492188, 0.087890625, 5.029296875, 0, -3.666992188],
                        [1.543945313, -0.0146484375, -2.979492188, 0.087890625, 5.029296875, 1, -1.399898911],
                        [1.543945313, -0.0146484375, -2.979492188, 0.087890625, 5.029296875, 2, -132.7954168],
                        [1.543945313, -0.0146484375, -2.979492188, 0.087890625, 5.029296875, 3, 19.77728738],
                        [0.0205078125, -0.0205078125, -0.0146484375, 0.146484375, 4.951171875, 0, -5.083007813],
                        [0.0205078125, -0.0205078125, -0.0146484375, 0.146484375, 4.951171875, 1, -1.399738693],
                        [0.0205078125, -0.0205078125, -0.0146484375, 0.146484375, 4.951171875, 2, -105.0444555],
                        [0.0205078125, -0.0205078125, -0.0146484375, 0.146484375, 4.951171875, 3, -3.999867264],
                        [0.0205078125, -0.0205078125, -0.0146484375, 0.146484375, 4.951171875, 4, 11.12800676],
                        [0.0205078125, -0.0205078125, -2.985351563, 0.146484375, 5.048828125, 0, -2.209960938],
                        [0.0205078125, -0.0205078125, -2.985351563, 0.146484375, 5.048828125, 1, -1.399738693],
                        [0.0205078125, -0.0205078125, -2.985351563, 0.146484375, 5.048828125, 2, -127.7925024],
                        [0.0205078125, -0.0205078125, -2.985351563, 0.146484375, 5.048828125, 3, 11.42213586],
                        [0.0146484375, -1.497070313, -2.276367188, 0.107421875, 2.548828125, 0, 1.102539063],
                        [1.497070313, -0.7646484375, -1.526367188, 0.107421875, 2.509765625, 0, -1.823242188],
                        [1.497070313, -0.7646484375, -1.526367188, 0.107421875, 2.509765625, 1, -1.334919357],
                        [1.497070313, -0.7646484375, -1.526367188, 0.107421875, 2.509765625, 2, -73.59329891],
                        [1.497070313, -0.7646484375, -1.526367188, 0.107421875, 2.509765625, 3, 9.073244934],
                        [0.0029296875, -2.997070313, -2.997070313, 0.009765625, 5.009765625, 0, 0.9716796875],
                        [0.7529296875, -1.502929688, -2.997070313, 0.009765625, 7.490234375, 0, -3.752929688],
                        [0.7529296875, -1.502929688, -2.997070313, 0.009765625, 7.490234375, 1, -1.14902153],
                        [0.7529296875, -1.502929688, -2.997070313, 0.009765625, 7.490234375, 2, -181.2965078],
                        [0.7529296875, -1.502929688, -2.997070313, 0.009765625, 7.490234375, 3, -5.984633727],
                        [0.7529296875, -1.502929688, -2.997070313, 0.009765625, 7.490234375, 4, 14.86249666],
                        [0.7412109375, -1.491210938, -0.0087890625, 0.126953125, 7.373046875, 0, -6.741210938],
                        [0.7412109375, -1.491210938, -0.0087890625, 0.126953125, 7.373046875, 1, -1.152759933],
                        [0.7412109375, -1.491210938, -0.0087890625, 0.126953125, 7.373046875, 2, -157.4182043],
                        [0.7412109375, -1.491210938, -0.0087890625, 0.126953125, 7.373046875, 3, -2.060180158],
                        [0.7412109375, -1.491210938, -0.0087890625, 0.126953125, 7.373046875, 4, 16.12132004],
                        [1.502929688, 2.592773438, 0.6767578125, 6.533203125, -1.923828125, 0, -9.381835938],
                        [1.502929688, 2.592773438, 0.6767578125, 6.533203125, -1.923828125, 1, -0.2262310028],
                        [1.502929688, 2.592773438, 0.6767578125, 6.533203125, -1.923828125, 2, -35.57202816],
                        [1.502929688, 2.592773438, 0.6767578125, 6.533203125, -1.923828125, 3, -2.129525428],
                        [1.502929688, 2.592773438, 0.6767578125, 6.533203125, -1.923828125, 4, 9.263431268],
                        [1.555664063, 2.774414063, -2.592773438, 0.224609375, 1.962890625, 0, -3.924804688],
                        [1.555664063, 2.774414063, -2.592773438, 0.224609375, 1.962890625, 1, -0.5442317963],
                        [1.555664063, 2.774414063, -2.592773438, 0.224609375, 1.962890625, 2, -95.90186214],
                        [1.555664063, 2.774414063, -2.592773438, 0.224609375, 1.962890625, 3, -6.125070529],
                        [1.555664063, 2.774414063, -2.592773438, 0.224609375, 1.962890625, 4, 4.826327724],
                        [1.555664063, 2.774414063, 2.592773438, -0.224609375, 1.962890625, 0, -8.661132813],
                        [1.555664063, 2.774414063, 2.592773438, -0.224609375, 1.962890625, 1, -0.5442317963],
                        [1.555664063, 2.774414063, 2.592773438, -0.224609375, 1.962890625, 2, -75.15967464],
                        [1.555664063, 2.774414063, 2.592773438, -0.224609375, 1.962890625, 3, -7.124047304],
                        [1.555664063, 2.774414063, 2.592773438, -0.224609375, 1.962890625, 4, 17.82354359],
                        [2.469726563, 2.528320313, -2.663085938, -7.529296875, 1.845703125, 0, 3.348632813],
                        [2.715820313, 0.0029296875, -2.569335938, -3.037109375, 2.939453125, 0, -0.0517578125],
                        [2.715820313, 0.0029296875, -2.569335938, -3.037109375, 2.939453125, 1, -1.307758713],
                        [2.715820313, 0.0029296875, -2.569335938, -3.037109375, 2.939453125, 2, -98.73279476],
                        [2.715820313, 0.0029296875, -2.569335938, -3.037109375, 2.939453125, 3, 17.17429903],
                        [2.469726563, 2.528320313, 2.663085938, -7.529296875, -1.845703125, 0, 1.713867188],
                        [2.938476563, 1.708007813, -0.1904296875, -1.025390625, 1.826171875, 0, -5.256835938],
                        [2.938476563, 1.708007813, -0.1904296875, -1.025390625, 1.826171875, 1, -1.065342331],
                        [2.938476563, 1.708007813, -0.1904296875, -1.025390625, 1.826171875, 2, -77.65552425],
                        [2.938476563, 1.708007813, -0.1904296875, -1.025390625, 1.826171875, 3, 5.27970673],
                        [2.715820313, 0.0029296875, 2.569335938, -6.962890625, 2.939453125, 0, -1.264648438],
                        [2.715820313, 0.0029296875, 2.569335938, -6.962890625, 2.939453125, 1, -0.9151805878],
                        [2.715820313, 0.0029296875, 2.569335938, -6.962890625, 2.939453125, 2, -78.17810726],
                        [2.715820313, 0.0029296875, 2.569335938, -6.962890625, 2.939453125, 3, 5.883202479],
                        [2.891601563, 2.217773438, 0.3837890625, -9.384765625, 3.603515625, 0, 0.2880859375],
                        [0.5654296875, 2.991210938, -1.526367188, -0.224609375, 0.537109375, 0, -2.342773438],
                        [0.5654296875, 2.991210938, -1.526367188, -0.224609375, 0.537109375, 1, -0.4053462982],
                        [0.5654296875, 2.991210938, -1.526367188, -0.224609375, 0.537109375, 2, -67.4576025],
                        [0.5654296875, 2.991210938, -1.526367188, -0.224609375, 0.537109375, 3, -11.41781517],
                        [0.5654296875, 2.991210938, -1.526367188, -0.224609375, 0.537109375, 4, 6.223440706],
                        [0.7236328125, 0.2255859375, 1.098632813, -8.837890625, 2.978515625, 0, 3.811523438],
                        [2.938476563, 1.291992188, 0.1904296875, -1.025390625, 1.826171875, 0, -5.221679688],
                        [2.938476563, 1.291992188, 0.1904296875, -1.025390625, 1.826171875, 1, -1.204014206],
                        [2.938476563, 1.291992188, 0.1904296875, -1.025390625, 1.826171875, 2, -73.21997738],
                        [2.938476563, 1.291992188, 0.1904296875, -1.025390625, 1.826171875, 3, 4.581538822],
                        [2.891601563, 0.7822265625, -0.3837890625, -9.384765625, 3.603515625, 0, 2.491210938],
                        [1.280273438, 2.903320313, -0.3369140625, -0.439453125, 9.482421875, 0, -12.88964844],
                        [1.280273438, 2.903320313, -0.3369140625, -0.439453125, 9.482421875, 1, -0.4614833832],
                        [1.280273438, 2.903320313, -0.3369140625, -0.439453125, 9.482421875, 2, -241.4439077],
                        [1.280273438, 2.903320313, -0.3369140625, -0.439453125, 9.482421875, 3, -4.215422393],
                        [1.280273438, 2.903320313, -0.3369140625, -0.439453125, 9.482421875, 4, 29.33357475],
                        [0.9814453125, 0.0615234375, 2.475585938, -7.841796875, 6.279296875, 0, -1.956054688],
                        [0.9814453125, 0.0615234375, 2.475585938, -7.841796875, 6.279296875, 1, -0.7846416473],
                        [0.9814453125, 0.0615234375, 2.475585938, -7.841796875, 6.279296875, 2, -132.6247244],
                        [0.9814453125, 0.0615234375, 2.475585938, -7.841796875, 6.279296875, 3, 4.022597458],
                        [0.5654296875, 2.991210938, 1.473632813, -0.224609375, 0.537109375, 0, -5.342773438],
                        [0.5654296875, 2.991210938, 1.473632813, -0.224609375, 0.537109375, 1, -0.4053462982],
                        [0.5654296875, 2.991210938, 1.473632813, -0.224609375, 0.537109375, 2, -55.29939938],
                        [0.5654296875, 2.991210938, 1.473632813, -0.224609375, 0.537109375, 3, -10.94159175],
                        [0.5654296875, 2.991210938, 1.473632813, -0.224609375, 0.537109375, 4, 6.375599953],
                        [1.719726563, 2.903320313, 0.3369140625, -0.439453125, 9.482421875, 0, -14.00292969],
                        [1.719726563, 2.903320313, 0.3369140625, -0.439453125, 9.482421875, 1, -0.4614833832],
                        [1.719726563, 2.903320313, 0.3369140625, -0.439453125, 9.482421875, 2, -240.9458609],
                        [1.719726563, 2.903320313, 0.3369140625, -0.439453125, 9.482421875, 3, 1.14853274],
                        [0.7236328125, 0.2255859375, -1.901367188, -8.837890625, 2.978515625, 0, 6.811523438],
                        [0.4013671875, 0.0615234375, 2.504882813, -8.310546875, 2.548828125, 0, 2.793945313],
                        [0.5302734375, 2.727539063, 0.1376953125, -1.083984375, 9.345703125, 0, -11.65722656],
                        [0.5302734375, 2.727539063, 0.1376953125, -1.083984375, 9.345703125, 1, -0.5616420746],
                        [0.5302734375, 2.727539063, 0.1376953125, -1.083984375, 9.345703125, 2, -230.9587393],
                        [0.5302734375, 2.727539063, 0.1376953125, -1.083984375, 9.345703125, 3, -3.988254441],
                        [0.5302734375, 2.727539063, 0.1376953125, -1.083984375, 9.345703125, 4, 20.19257367],
                        [2.200195313, 2.551757813, -2.282226563, 8.076171875, -0.185546875, 0, -10.36035156],
                        [2.200195313, 2.551757813, -2.282226563, 8.076171875, -0.185546875, 1, -0.02425804138],
                        [2.200195313, 2.551757813, -2.282226563, 8.076171875, -0.185546875, 2, -69.47617245],
                        [2.200195313, 2.551757813, -2.282226563, 8.076171875, -0.185546875, 3, 5.908934768]],
                       dtype=np.double)
