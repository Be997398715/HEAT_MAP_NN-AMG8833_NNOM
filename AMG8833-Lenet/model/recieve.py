import serial
from time import sleep
from threading import Thread
import struct
import binascii
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt


def receive_Data(serial,filepath):
	counter=0
	while True:
		rec = serial.read(1)
		if rec:
			if(str(rec.decode())=="F"):
				buf = ''
				buf1 = []
				filename = filepath+"/"+str(counter)+".bmp"
				im = Image.new("L", (8, 8))
				while True:
					res = serial.read(1)
					if(str(res.decode('utf-8'))=='E'):
						buf1 = buf.split(',')
						del(buf1[-1])
						print(len(buf1))
						print(buf1)
						for i in range(len(buf1)):
							im.putpixel((int(i/8), int(i%8)),(int(buf1[i])))
						im.save(filename)
						#im.show()
						print('save picture successfully!')
						counter=counter+1
						print(counter)
						break
					else:
						buf = buf + str(res.decode('utf-8'))



if __name__ == '__main__':
	serial_1 = serial.Serial('COM4', 115200, timeout=0.5)  #/dev/ttyUSB0
	filepath = r'data/0'
	if serial_1.isOpen() :
		print("open success")
	else :
		print("open failed")
	receive_Data(serial_1,filepath)

			