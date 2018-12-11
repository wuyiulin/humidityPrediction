#coding:utf-8

#fname1 = r'C:\Users\Franky\Desktop\gradeProgram\true_data.txt'
fname1 = r'F:\gradeProram\realData.txt'
fnameB = r'F:\gradeProram\backupData.txt'
import numpy as np
from sklearn import linear_model
import linecache
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures
import serial

row = 270
#row3 = row/3
Hour = 0
Minute = 0
Sec = 0
X_axix = np.zeros((270), dtype=np.float64) ##
Y_axix = np.zeros((270), dtype=np.float64) ##
Line_amount = 0


#從NodeMCU獲得資料
while (1):
    ser = serial.Serial('COM3')
    ser.baudrate = 9600 
    ser.port = 'COM3' 
    hByte = ser.readline()
    #Hstr = Hstr.encode('base64','strict')
    hByte = str(hByte, encoding='utf-8')
    Htxt_f = open(fname1,'r')
    HtxtCache = Htxt_f.readlines()
    Htxt_f.close()
    Htxt_f = open(fname1,'w')
    Htxt_f.writelines(HtxtCache)
    Htxt_f.close()
    Htxt_f = open(fname1,'a')
    Htxt_f.write(hByte)
    Htxt_f.close()
    HtxtB_f = open(fnameB, 'a')
    HtxtB_f.write(hByte)
    HtxtB_f.close()

    print(str(hByte))
    ser.close()
    """
    #把新的三行資料加進來
    old_f = open( fname1, 'r')
    write_old_line = old_f.readlines()
    new_f = open( fname1, 'w')
    new_f.write(time.strftime('%H\n%M\n%S\n', time.localtime()))
    new_f.writelines(write_old_line)
    new_f.close()
    Line_amount = len(open(fname1,'r').readlines())
    print('這是未處理前的行數:'+str(Line_amount))
    """



    #多餘行數處理區
    while Line_amount > (row-1):
        with open(fname1, 'r') as old_file:
            with open(fname1, 'r+') as new_file:
                current_line = 1
                while current_line < (row+1):
                     old_file.readline()
                     current_line += 1
                seek_point = old_file.tell()
                new_file.seek(seek_point, 0)
                old_file.readline()
                next_line = old_file.readline()
                while next_line:
                    new_file.write(next_line)
                    next_line = old_file.readline()
                new_file.truncate()
            new_file.close()
        Line_amount = Line_amount - 1
        #old_file.close()
    Line_amount = len(open(fname1,'r').readlines())
    print('這是已處理後的行數:'+str(Line_amount))



    #開檔計算數據並預測
    with open(fname1, 'r') as old_file:
        #f = open(fname2,'w+')
        #hour = 0
        #minute = 0
        #sec = 0
        date_List = old_file.readlines()
        #date_List.reverse()
        #print(old_file.readlines())
        #print(date_List)
        #str = ';'.join(list)
        Count = 0
        Count = int(Count)
        print(date_List[0])
        while Count < (row):
            data_Sum = 0
            Hour = float(date_List[Count])
            #print('這是小時:'+str(Hour)) ##
            #Minute = float(date_List[Count + 1])*60 ##
            #Sec = float(date_List[Count + 2]) ##
            data_Sum = Hour #+ Minute + Sec ##
            X_axix[int(Count)] = (int(Count)) ##
            Y_axix[int(Count)] = data_Sum ##
            print('這是日期座標：'+ str(X_axix[int(Count)]))
            print('這是時間總和：'+ str(Y_axix[int(Count)]))
            #print(Count)
            Count = int(Count) + 1 ##
        #print(Hour)
        #f.write(str)
        #print(f.readlines())
        Coe_X_axix = X_axix.reshape((int(Count), 1))
        Coe_Y_axix = Y_axix.reshape((int(Count), 1))

        minX = min(X_axix)
        maxX = max(X_axix)
        '''
        X = np.arange(minX,maxX).reshape((int(Count/3), 1))
        print('這是X: '+str(X))
        '''
        #Y_axix.reshape((1,-1)
        print(list(Coe_X_axix))
        print(list(Coe_Y_axix))



        #選取分析方法
        #regr = linear_model.LinearRegression()
        regr = linear_model.LogisticRegression()
        '''
        poly_reg = PolynomialFeatures(degree = 2)
        X_poly = poly_reg.fit_transform(X_axix)
        '''
        regr.fit((Coe_X_axix), list(Coe_Y_axix))
        #regr.fit((X_poly), (Y_axix))


        print('Coefficients: \n', regr.coef_)
        #print(Count)
        Y_axix_pred = regr.predict(Coe_X_axix)
        R_squared = regr.score((Coe_X_axix), list(Coe_Y_axix))
        print(Y_axix_pred)
        H = int(regr.predict(Count)) ##
        #M = int((regr.predict(Count)/60)%60) ##
        #S = int(regr.predict(Count)%60) ##
        print('預測下個時間點的溼度: '+str(H))##+':'+str(M)+':'+str(S)) ##
        print('R_squared: '+str(R_squared))



        #畫圖區
        plt.figure()
        plt.xlim((0, 270)) ##
        plt.ylim((70, 100)) ##
        T = np.arctan2(Y_axix,X_axix)
        plt.scatter(X_axix, Y_axix, s=100, c=T, alpha=.5)
        plt.plot(X_axix, Y_axix, '-')
        plt.xlabel('nearly 90day')
        plt.ylabel('Total Sec')
        plt.plot(X_axix, Y_axix_pred, 'blue', 20)
        #plt.plot(X, regr.predict(poly_reg.fit_transform(X)), 'blue', 20)
        x_trick = np.arange(0, 270, 30) ##
        y_trick = np.arange(70, 100, 5) ##
        plt.xticks(x_trick)
        plt.yticks(y_trick)

        #收集數據，暫時關閉繪圖區
        '''
        plt.show(block=False)
        plt.pause(5)
        plt.close(1)
        '''
        




    time.sleep(120)
    