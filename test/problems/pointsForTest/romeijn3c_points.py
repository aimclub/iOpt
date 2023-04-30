import numpy as np

test_points = np.array(
    [[3.50634765625, 1.50537109375, 2, 0.01171875],
     [-2.99365234375, 1.50537109375, 2, -6.48828125],
     [-2.99365234375, 1.50537109375, 3, -5.25979447365],
     [-2.99365234375, 1.50537109375, 4, -137.770705709],
     [-2.99365234375, 1.50537109375, 1, 0.442135494892],
     [9.99365234375, 1.49462890625, 2, 6.48828125],
     [0.25634765625, -1.24462890625, 2, -5.98828125],
     [0.25634765625, -1.24462890625, 3, -1.29275345802],
     [0.25634765625, -1.24462890625, 4, -2.39433347851],
     [0.25634765625, -1.24462890625, 1, 0.49024777988],
     [0.25634765625, 4.25537109375, 2, -0.48828125],
     [0.25634765625, 4.25537109375, 3, -17.8518354893],
     [0.25634765625, 4.25537109375, 4, -28.8888647285],
     [0.25634765625, 4.25537109375, 1, 3.99079562173],
     [0.25634765625, -3.99462890625, 2, -8.73828125],
     [0.25634765625, -3.99462890625, 3, -15.7007124424],
     [0.25634765625, -3.99462890625, 4, -25.4470678535],
     [0.25634765625, -3.99462890625, 1, 0.25458403192],
     [-1.36865234375, -2.61962890625, 2, -8.98828125],
     [-1.36865234375, -2.61962890625, 3, -8.23110795021],
     [-1.36865234375, -2.61962890625, 4, -23.7987900403],
     [-1.36865234375, -2.61962890625, 1, 0.283941976973],
     [-2.99365234375, -2.61962890625, 2, -10.61328125],
     [-2.99365234375, -2.61962890625, 3, -9.85610795021],
     [-2.99365234375, -2.61962890625, 4, -145.124807271],
     [-2.99365234375, -2.61962890625, 1, 0.227416890304],
     [0.24365234375, 1.46240234375, 2, -3.2939453125],
     [0.24365234375, 1.46240234375, 3, -1.89496827126],
     [0.24365234375, 1.46240234375, 4, -3.34946909279],
     [0.24365234375, 1.46240234375, 1, 1.0668839404],
     [-2.18115234375, -3.30712890625, 2, -10.48828125],
     [-2.18115234375, -3.30712890625, 3, -13.1182539463],
     [-2.18115234375, -3.30712890625, 4, -69.3827119705],
     [-2.18115234375, -3.30712890625, 1, 0.225670967726],
     [1.95751953125, -2.72705078125, 2, -5.76953125],
     [1.95751953125, -2.72705078125, 3, -5.47928643227],
     [1.95751953125, -2.72705078125, 4, 25.6060367409],
     [1.77978515625, -3.98388671875, 2, -7.2041015625],
     [1.77978515625, -3.98388671875, 3, -14.0915682316],
     [1.77978515625, -3.98388671875, 4, 2.79438514777],
     [1.89404296875, -1.33056640625, 2, -4.4365234375],
     [1.89404296875, -1.33056640625, 3, 0.123636007309],
     [1.00537109375, -3.27490234375, 2, -7.26953125],
     [1.00537109375, -3.27490234375, 3, -9.71961426735],
     [1.00537109375, -3.27490234375, 4, -12.078976667],
     [1.00537109375, -3.27490234375, 1, 0.318029036016],
     [1.11962890625, -1.97509765625, 2, -5.85546875],
     [1.11962890625, -1.97509765625, 3, -2.78138184547],
     [1.11962890625, -1.97509765625, 4, 0.776042610523],
     [-2.18115234375, -3.99462890625, 2, -11.17578125],
     [-2.18115234375, -3.99462890625, 3, -18.1382124424],
     [-2.18115234375, -3.99462890625, 4, -77.4146455643],
     [-2.18115234375, -3.99462890625, 1, 0.20077150461],
     [0.23095703125, -2.63037109375, 2, -7.3994140625],
     [0.23095703125, -2.63037109375, 3, -6.68789505959],
     [0.23095703125, -2.63037109375, 4, -11.0085657768],
     [0.23095703125, -2.63037109375, 1, 0.344590438417],
     [-1.63525390625, 0.00146484375, 2, -6.6337890625],
     [-1.63525390625, 0.00146484375, 3, -1.63525605202],
     [-1.63525390625, 0.00146484375, 4, -21.8638006174],
     [-1.63525390625, 0.00146484375, 1, 0.463197620783],
     [1.81787109375, -3.23193359375, 2, -6.4140625],
     [1.81787109375, -3.23193359375, 3, -8.62752366066],
     [1.81787109375, -3.23193359375, 4, 13.324555239],
     [1.77978515625, -1.88916015625, 2, -5.109375],
     [1.77978515625, -1.88916015625, 3, -1.78914093971],
     [1.77978515625, -1.88916015625, 4, 22.4782688148],
     [2.09716796875, -0.15966796875, 2, -3.0625],
     [2.09716796875, -0.15966796875, 3, 2.07167410851],
     [1.81787109375, -0.96533203125, 2, -4.1474609375],
     [1.81787109375, -0.96533203125, 3, 0.886005163193],
     [2.22412109375, 1.44091796875, 2, -1.3349609375],
     [2.22412109375, 1.44091796875, 3, 0.147876501083],
     [0.89111328125, -0.47119140625, 2, -4.580078125],
     [0.89111328125, -0.47119140625, 3, 0.669091939926],
     [4.36962890625, 5.26513671875, 2, 4.634765625],
     [6.74365234375, -1.25537109375, 2, 0.48828125],
     [6.75634765625, -3.99462890625, 2, -2.23828125],
     [6.75634765625, -3.99462890625, 3, -9.2007124424],
     [6.75634765625, -3.99462890625, 4, 1516.54538608],
     [2.84619140625, -1.96435546875, 2, -4.1181640625],
     [2.84619140625, -1.96435546875, 3, -1.01250100136],
     [2.84619140625, -1.96435546875, 4, 109.108307436],
     [-2.68896484375, 0.27001953125, 2, -7.4189453125],
     [-2.68896484375, 0.27001953125, 3, -2.76187539101],
     [-2.68896484375, 0.27001953125, 4, -97.3298876949],
     [-2.68896484375, 0.27001953125, 1, 0.388302633294],
     [0.29443359375, -1.94287109375, 2, -6.6484375],
     [0.29443359375, -1.94287109375, 3, -3.48031449318],
     [0.29443359375, -1.94287109375, 4, -5.91197301794],
     [0.29443359375, -1.94287109375, 1, 0.410953461256],
     [-2.58740234375, -3.65087890625, 2, -11.23828125],
     [-2.58740234375, -3.65087890625, 3, -15.9163191319],
     [-2.58740234375, -3.65087890625, 4, -107.935043858],
     [-2.58740234375, -3.65087890625, 1, 0.202882122303],
     [0.42138671875, -0.58935546875, 2, -5.16796875],
     [0.42138671875, -0.58935546875, 3, 0.0740468502045],
     [-2.20654296875, -1.91064453125, 2, -9.1171875],
     [-2.20654296875, -1.91064453125, 3, -5.85710549355],
     [-2.20654296875, -1.91064453125, 4, -59.557333716],
     [-2.20654296875, -1.91064453125, 1, 0.288011213583],
     [-0.56884765625, -3.37158203125, 2, -8.9404296875],
     [-0.56884765625, -3.37158203125, 3, -11.9364130497],
     [-0.56884765625, -3.37158203125, 4, -19.1084650281],
     [-0.56884765625, -3.37158203125, 1, 0.268361070406],
     [-0.58154296875, -1.91064453125, 2, -7.4921875],
     [-0.58154296875, -1.91064453125, 3, -4.23210549355],
     [-0.58154296875, -1.91064453125, 4, -6.8242665909],
     [-0.58154296875, -1.91064453125, 1, 0.367322738986],
     [1.39892578125, -2.89892578125, 2, -6.5],
     [1.39892578125, -2.89892578125, 3, -7.00484490395],
     [1.39892578125, -2.89892578125, 4, 0.242409099103],
     [1.23388671875, 0.69970703125, 2, -3.06640625],
     [1.23388671875, 0.69970703125, 3, 0.744296789169],
     [2.79541015625, -3.36083984375, 2, -5.5654296875],
     [2.79541015625, -3.36083984375, 3, -8.49983429909],
     [2.79541015625, -3.36083984375, 4, 91.148727563],
     [1.43701171875, -1.54541015625, 2, -5.1083984375],
     [1.43701171875, -1.54541015625, 3, -0.951280832291],
     [1.43701171875, -1.54541015625, 4, 11.0158971691],
     [0.67529296875, -2.35107421875, 2, -6.67578125],
     [0.67529296875, -2.35107421875, 3, -4.85225701332],
     [0.67529296875, -2.35107421875, 4, -7.30434246885],
     [0.67529296875, -2.35107421875, 1, 0.387658967809],
     [-2.99365234375, -3.65087890625, 2, -11.64453125],
     [-2.99365234375, -3.65087890625, 3, -16.3225691319],
     [-2.99365234375, -3.65087890625, 4, -155.471145162],
     [-2.99365234375, -3.65087890625, 1, 0.193289315665],
     [-2.10498046875, 0.91455078125, 2, -6.1904296875],
     [-2.10498046875, 0.91455078125, 3, -2.94138360023],
     [-2.10498046875, 0.91455078125, 4, -47.9734849956],
     [-2.10498046875, 0.91455078125, 1, 0.499804006513],
     [-0.32763671875, 0.08740234375, 2, -5.240234375],
     [-0.32763671875, 0.08740234375, 3, -0.335275888443],
     [-0.32763671875, 0.08740234375, 4, -0.188074831828],
     [-0.32763671875, 0.08740234375, 1, 0.626440707082],
     [-2.58740234375, -0.86865234375, 2, -8.4560546875],
     [-2.58740234375, -0.86865234375, 3, -3.34195923805],
     [-2.58740234375, -0.86865234375, 4, -87.8160680276],
     [-2.58740234375, -0.86865234375, 1, 0.325276980842],
     [-2.18115234375, -2.63037109375, 2, -9.8115234375],
     [-2.18115234375, -2.63037109375, 3, -9.10000443459],
     [-2.18115234375, -2.63037109375, 4, -62.9535127518],
     [-2.18115234375, -2.63037109375, 1, 0.25406045327],
     [-2.96826171875, -1.97509765625, 2, -9.943359375],
     [-2.96826171875, -1.97509765625, 3, -6.86927247047],
     [-2.96826171875, -1.97509765625, 4, -137.002118714],
     [-2.96826171875, -1.97509765625, 1, 0.25325591471],
     [-0.51806640625, -3.96240234375, 2, -9.48046875],
     [-0.51806640625, -3.96240234375, 3, -16.21869874],
     [-0.51806640625, -3.96240234375, 4, -25.8162382041],
     [-0.51806640625, -3.96240234375, 1, 0.239451239744],
     [-1.38134765625, -1.92138671875, 2, -8.302734375],
     [-1.38134765625, -1.92138671875, 3, -5.07307457924],
     [-1.38134765625, -1.92138671875, 4, -19.0856578323],
     [-1.38134765625, -1.92138671875, 1, 0.325887810515],
     [0.28173828125, -3.38232421875, 2, -8.1005859375],
     [0.28173828125, -3.38232421875, 3, -11.1583788395],
     [0.28173828125, -3.38232421875, 4, -18.1923704574],
     [0.28173828125, -3.38232421875, 1, 0.291375943156],
     [0.71337890625, -0.97607421875, 2, -5.2626953125],
     [0.71337890625, -0.97607421875, 3, -0.239341974258],
     [0.71337890625, -0.97607421875, 4, 0.29087297481],
     [-0.42919921875, 3.03076171875, 2, -2.3984375],
     [-0.42919921875, 3.03076171875, 3, -9.61471581459],
     [-0.42919921875, 3.03076171875, 4, -15.09214472],
     [-0.42919921875, 3.03076171875, 1, 1.44986676],
     [0.70068359375, -0.66455078125, 2, -4.9638671875],
     [0.70068359375, -0.66455078125, 3, 0.25905585289],
     [-0.58154296875, -2.69482421875, 2, -8.2763671875],
     [-0.58154296875, -2.69482421875, 3, -7.84362053871],
     [-0.58154296875, -2.69482421875, 4, -12.6026906632],
     [-0.58154296875, -2.69482421875, 1, 0.30892909956],
     [2.83349609375, 0.69970703125, 2, -1.466796875],
     [2.83349609375, 0.69970703125, 3, 2.34390616417],
     [-1.53369140625, 1.63427734375, 2, -4.8994140625],
     [-1.53369140625, 1.63427734375, 3, -4.20455384254],
     [-1.53369140625, 1.63427734375, 4, -22.3111960707],
     [-1.53369140625, 1.63427734375, 1, 0.675193634507],
     [2.82080078125, -1.25537109375, 2, -3.4345703125],
     [2.82080078125, -1.25537109375, 3, 1.24484419823],
     [1.13232421875, -3.01708984375, 2, -6.884765625],
     [1.13232421875, -3.01708984375, 3, -7.97050690651],
     [1.13232421875, -3.01708984375, 4, -7.30543625003],
     [1.13232421875, -3.01708984375, 1, 0.34226039517],
     [6.10888671875, 1.73095703125, 2, 2.83984375],
     [4.26806640625, 1.48388671875, 2, 0.751953125],
     [3.81103515625, -0.98681640625, 2, -2.17578125],
     [3.81103515625, -0.98681640625, 3, 2.83722853661],
     [5.04248046875, 0.44189453125, 2, 0.484375],
     [5.13134765625, -0.13818359375, 2, -0.0068359375],
     [5.13134765625, -0.13818359375, 3, 5.11225295067],
     [4.05224609375, -0.10595703125, 2, -1.0537109375],
     [4.05224609375, -0.10595703125, 3, 4.04101920128],
     [-1.36865234375, -1.99658203125, 2, -8.365234375],
     [-1.36865234375, -1.99658203125, 3, -5.35499215126],
     [-1.36865234375, -1.99658203125, 4, -19.197004762],
     [-1.36865234375, -1.99658203125, 1, 0.321559224823],
     [-0.69580078125, -0.97607421875, 2, -6.671875],
     [-0.69580078125, -0.97607421875, 3, -1.64852166176],
     [-0.69580078125, -0.97607421875, 4, -3.20867393187],
     [-0.69580078125, -0.97607421875, 1, 0.447622818857],
     [-2.79052734375, -3.82275390625, 2, -11.61328125],
     [-2.79052734375, -3.82275390625, 3, -17.4039747715],
     [-2.79052734375, -3.82275390625, 4, -132.031295971],
     [-2.79052734375, -3.82275390625, 1, 0.192727994554],
     [1.52587890625, -1.25537109375, 2, -4.7294921875],
     [1.52587890625, -1.25537109375, 3, -0.0500776767731],
     [1.52587890625, -1.25537109375, 4, 15.2420378612],
     [-2.34619140625, 2.24658203125, 2, -5.099609375],
     [-2.34619140625, 2.24658203125, 3, -7.39332222939],
     [-2.34619140625, 2.24658203125, 4, -72.6498009709],
     [-2.34619140625, 2.24658203125, 1, 0.601674121429],
     [0.23095703125, -1.93212890625, 2, -6.701171875],
     [0.23095703125, -1.93212890625, 3, -3.50216507912],
     [0.23095703125, -1.93212890625, 4, -5.91139780802],
     [0.23095703125, -1.93212890625, 1, 0.408670702033],
     [-1.76220703125, -3.66162109375, 2, -10.423828125],
     [-1.76220703125, -3.66162109375, 3, -15.1696760654],
     [-1.76220703125, -3.66162109375, 4, -48.8135066025],
     [-1.76220703125, -3.66162109375, 1, 0.222767326802],
     [-2.71435546875, -0.33154296875, 2, -8.0458984375],
     [-2.71435546875, -0.33154296875, 3, -2.82427620888],
     [-2.71435546875, -0.33154296875, 4, -100.169004703],
     [-2.71435546875, -0.33154296875, 1, 0.348417004759],
     [3.04931640625, -0.89013671875, 2, -2.8408203125],
     [3.04931640625, -0.89013671875, 3, 2.25697302818],
     [1.84326171875, 1.05419921875, 2, -2.1025390625],
     [1.84326171875, 1.05419921875, 3, 0.731925725937],
     [8.07666015625, 0.24853515625, 2, 3.3251953125],
     [0.97998046875, -2.24365234375, 2, -6.263671875],
     [0.97998046875, -2.24365234375, 3, -4.05399537086],
     [0.97998046875, -2.24365234375, 4, -3.34868270496],
     [0.97998046875, -2.24365234375, 1, 0.412020654435],
     [-1.76220703125, -2.97412109375, 2, -9.736328125],
     [-1.76220703125, -2.97412109375, 3, -10.6076033115],
     [-1.76220703125, -2.97412109375, 4, -41.5141901963],
     [-1.76220703125, -2.97412109375, 1, 0.252289327297],
     [-2.60009765625, -2.98486328125, 2, -10.5849609375],
     [-2.60009765625, -2.98486328125, 3, -11.509506464],
     [-2.60009765625, -2.98486328125, 4, -102.144956808],
     [-2.60009765625, -2.98486328125, 1, 0.226010996937],
     [-2.57470703125, -2.30810546875, 2, -9.8828125],
     [-2.57470703125, -2.30810546875, 3, -7.90205788612],
     [-2.57470703125, -2.30810546875, 4, -93.8639230691],
     [-2.57470703125, -2.30810546875, 1, 0.254173371745],
     [-0.12451171875, -3.65087890625, 2, -8.775390625],
     [-0.12451171875, -3.65087890625, 3, -13.4534285069],
     [-0.12451171875, -3.65087890625, 4, -21.3359184915],
     [-0.12451171875, -3.65087890625, 1, 0.26471866929],
     [-1.55908203125, -0.84716796875, 2, -7.40625],
     [-1.55908203125, -0.84716796875, 3, -2.27677559853],
     [-1.55908203125, -0.84716796875, 4, -20.0968998909],
     [-1.55908203125, -0.84716796875, 1, 0.394352415834],
     [-1.02587890625, -3.66162109375, 2, -9.6875],
     [-1.02587890625, -3.66162109375, 3, -14.4333479404],
     [-1.02587890625, -3.66162109375, 4, -26.8502664731],
     [-1.02587890625, -3.66162109375, 1, 0.241402149017],
     [-2.61279296875, -1.55615234375, 2, -9.1689453125],
     [-2.61279296875, -1.55615234375, 3, -5.03440308571],
     [-2.61279296875, -1.55615234375, 4, -93.0581764288],
     [-2.61279296875, -1.55615234375, 1, 0.287436348396],
     [3.49365234375, 4.24462890625, 2, 2.73828125],
     [1.20849609375, 2.35400390625, 2, -1.4375],
     [1.20849609375, 2.35400390625, 3, -4.33283829689],
     [1.20849609375, 2.35400390625, 4, -0.0413170286687],
     [1.20849609375, 2.35400390625, 1, 1.94520241614],
     [1.84326171875, 3.70751953125, 2, 0.55078125],
     [0.70068359375, -3.66162109375, 2, -7.9609375],
     [0.70068359375, -3.66162109375, 3, -12.7067854404],
     [0.70068359375, -3.66162109375, 4, -19.7319211324],
     [0.70068359375, -3.66162109375, 1, 0.283698135163],
     [-1.06396484375, -2.99560546875, 2, -9.0595703125],
     [-1.06396484375, -2.99560546875, 3, -10.0376169682],
     [-1.06396484375, -2.99560546875, 4, -20.379997135],
     [-1.06396484375, -2.99560546875, 1, 0.273853877659],
     [3.63330078125, 0.57080078125, 2, -0.7958984375],
     [3.63330078125, 0.57080078125, 3, 3.30748724937],
     [1.62744140625, -0.50341796875, 2, -3.8759765625],
     [1.62744140625, -0.50341796875, 3, 1.37401175499],
     [-2.79052734375, -3.99462890625, 2, -11.78515625],
     [-2.79052734375, -3.99462890625, 3, -18.7475874424],
     [-2.79052734375, -3.99462890625, 4, -134.181076244],
     [-2.79052734375, -3.99462890625, 1, 0.187583818852],
     [0.91650390625, -0.73974609375, 2, -4.8232421875],
     [0.91650390625, -0.73974609375, 3, 0.369279623032],
     [4.10302734375, 2.31103515625, 2, 1.4140625],
     [-2.11767578125, 0.36669921875, 2, -6.7509765625],
     [-2.11767578125, 0.36669921875, 3, -2.25214409828],
     [-2.11767578125, 0.36669921875, 4, -47.6992714966],
     [-2.11767578125, 0.36669921875, 1, 0.448411211813],
     [-0.92431640625, -2.29736328125, 2, -8.2216796875],
     [-0.92431640625, -2.29736328125, 3, -6.20219445229],
     [-0.92431640625, -2.29736328125, 4, -12.3931034823],
     [-0.92431640625, -2.29736328125, 1, 0.321772224918],
     [-1.80029296875, -2.29736328125, 2, -9.09765625],
     [-1.80029296875, -2.29736328125, 3, -7.07817101479],
     [-1.80029296875, -2.29736328125, 4, -37.6188454725],
     [-1.80029296875, -2.29736328125, 1, 0.284776716099],
     [0.62451171875, -2.90966796875, 2, -7.28515625],
     [0.62451171875, -2.90966796875, 3, -7.84165596962],
     [0.62451171875, -2.90966796875, 4, -12.3280239647],
     [0.62451171875, -2.90966796875, 1, 0.335766077548],
     [-1.81298828125, -1.56689453125, 2, -8.3798828125],
     [-1.81298828125, -1.56689453125, 3, -4.26814675331],
     [-1.81298828125, -1.56689453125, 4, -33.7240497565],
     [-1.81298828125, -1.56689453125, 1, 0.326882841504],
     [-2.19384765625, -3.66162109375, 2, -10.85546875],
     [-2.19384765625, -3.66162109375, 3, -15.6013166904],
     [-2.19384765625, -3.66162109375, 4, -74.2465382281],
     [-2.19384765625, -3.66162109375, 1, 0.212061343021],
     [-0.98779296875, -0.52490234375, 2, -6.5126953125],
     [-0.98779296875, -0.52490234375, 3, -1.26331543922],
     [-0.98779296875, -0.52490234375, 4, -5.25995656324],
     [-0.98779296875, -0.52490234375, 1, 0.470939019108]
     ], dtype=np.double)
