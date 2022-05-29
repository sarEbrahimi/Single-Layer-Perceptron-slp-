import numpy as np
import matplotlib.pyplot as plt
import random

#----------------------------------------------------------------LETS START
class perceptron(object):

    # -----------------------------------------------------------STEP 1
    # for use of samples . get them prepared
    def pre_sample(self):
        #target = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        sample1 = [0, 2]
        sample2 = [3, 4]
        sample3 = [2, 6]
        sample4 = [1, -2]
        sample5 = [4, -3]
        sample6 = [8, 6]
        sample7 = [5, 2]
        sample8 = [9, 3]
        sample9 = [6, 9]
        sample10 = [7, -1]
        sample = [sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8, sample9, sample10]
        return sample


     #illustrate sample on plot
    def show_sample(self):
        # i wanna see my sample on plot
        class1_x = np.array([0, 3, 2, 1, 4])
        class1_y = np.array([2, 4, 6, -2, -3])
        plt.scatter(class1_x, class1_y, color='blue')

        class2_x = np.array([8, 5, 9, 6, 7])
        class2_y = np.array([6, 2, 3, 9, -1])
        plt.scatter(class2_x, class2_y, color='red')

        plt.title('samples')
        plt.show()

    #------------------------------------------------------STEP 2
    #activation function
    # hardlim function
    def hardlim(self, n):
        return 1 if n >= 0 else  0

    # generate weight 1*2
    def gen_weight(self):
        weight = []
        for i in range(2):
            w = round(random.random(), 1)
            weight.append(w)  # 1*2
        print('weight is : ', weight)
        return weight



    def line_function(self,weight,bias):
        # line function
        # w1.x + w2.y + b = 0
        # we use flag to could swich between tow functions
        w1, w2 = weight
        flag = 0
        x = random.sample(range(0, 9), 9)
        # convert bias float to vector
        b = []
        y = []
        for i in range(0, 9):
            b.append(bias)

        for i in range(0, 9):
            try:

                a = -(w1 * x[i]) / w2
                a += b[i]
                y.append(a)
                flag = 1

            except:
                a = -(w2 * x[i]) / w1
                a += b[i]
                y.append(a)
                flag = 0

        return flag,x,y


    #------------------------------------------------------------------STEP 3
    #run an input through the perceptron and return an output
    def prediction(self):
        # x*w + b
        # a = hardlim(w * x + b)
        # error = target - a(output of hardlim function)
        weight = self.gen_weight()
        sample = self.pre_sample()
        bias = 0.5
        target = [0, 0, 0, 0, 0 , 1, 1, 1, 1, 1]
        class1_x = np.array([0, 3, 2, 1, 4])
        class1_y = np.array([2, 4, 6, -2, -3])
        class2_x = np.array([8, 5, 9, 6, 7])
        class2_y = np.array([6, 2, 3, 9, -1])

        flag_error = 1
        while (flag_error):
            flag_error = 0
            for i in range(len(sample)):


                print('***********the ', i + 1, 'th iteration************')
                s = np.matmul(weight, sample[i])
                s += [bias]
                a = self.hardlim(s)
                print('the sum of x*w+b is: : ', a)
                if a != target[i]:
                    e = target[i] - a
                    flag_error = 1
                    weight[0] = weight[0] + 0.3 * (sample[i][0] * e)
                    weight[1] = weight[1] + 0.3 * (sample[i][1] * e)
                    print('the updated WEIGHT is :', weight)
                    bias = bias + e
                    plt.scatter(sample[i][0], sample[i][1])
                    print('the new BIAS is : ', bias)

                else:
                    print('the real class is :', target[i])
                    print('the output is : ', a)
                    print('for this sample prediction was ok')

                x1, x2 = sample[i]

                flag, x, y = self.line_function(weight, bias)

                plt.scatter(class1_x, class1_y, color='blue', marker='s')
                plt.scatter(class2_x, class2_y, color='red', marker='^')
                plt.scatter(x1, x2, color='green', marker="8")
                if flag == 1:
                    plt.plot(x, y, color='green')
                else:
                    plt.plot(y, x, color='green')
                plt.pause(0.05)

        plt.clear()
        plt.show()











perceptron = perceptron()
perceptron.prediction()


