# HEAT_MAP_NN-AMG8833_NNOM
Mainly using AMG8833,  RT-Thread and NNOM libs to run heat-map recognition and setup neural network on STM32F4

(1). Introduction:
This project is my first attempt to do a neural network recognizing heat-map on stm32f4.
Main references are: 
1. NNOM : https://github.com/majianjia/nnom
2. RE-Thread : https://www.rt-thread.org/document/site/tutorial/quick-start/introduction/introduction/

(2). instructions:
1. AMG883-Lenet directory is using python3 and keras to train and test amg8833 heat-map data collected by recieve.py.
   Trainning is in amg8833_lenet-5.py. Some function is in utils.py.
2. stm32f407-NNOM-AMG8833-lenet.rar file includes mainly stm32 code. Also needs some libs like CMSIS-NN(version>1.8) and Rt-    Thread(version>5.2).
3. This is the first version, later I'd like to upload and modify some files.

(3). Results:
I trained 3 types of gestures recognition : None--0, one finger--1, two finger--2, it's about 70-80% accuracy in test, and FPS is 10.


Any questions please put issues or connect : 997398715(QQ)
