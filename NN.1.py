import numpy as np
import matplotlib.pyplot as plt
import random

# ----------------------------------------------------------------LETS START
class perceptron(object):

    # -----------------------------------------------------------STEP 1
    # for use of samples . get them prepared
    def pre_sample(self, sample):
        """takes in data , return a trained neural."""
        return sample

    def show_sample(self):
        """illustrate sample on plot."""
        class1_x = sample1[:,0]
        class1_y = sample1[:,1]
        plt.scatter(class1_x, class1_y, color='blue')

        class2_x = sample2[:,0]
        class2_y = sample2[:,1]
        plt.scatter(class2_x, class2_y, color='red')

        plt.title('Data Samples')
        plt.show()

    # ------------------------------------------------------STEP 2
    # activation function
    # hardlim function
    def hardlim(self, n):
        """take in n , return exit of function."""
        return 1 if n >= 0 else  0

    # generate weight 1*2
    def gen_weight(self):
        """generate weight in [0,1]."""
        weight = []
        for i in range(2):
            w = round(random.random(), 1)
            weight.append(w)  # 1*2
        print('weight is : ', weight)
        return weight

    def line_function(self, weight, bias):
        """take in w and b, calcute line function(boundary)."""
        # line function
        # w1.x + w2.y + b = 0
        # we use flag to could swich between tow functions for error in denominator
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

        return flag, x, y

    # ------------------------------------------------------------------STEP 3
    # run an input through the perceptron and return an output
    def prediction(self,sample1 , sample2 ,target):
        """
        generate weight
        calcute error
        use hardlim to get output

        """
        # x*w + b
        # a = hardlim(w * x + b)
        # error = target - a(output of hardlim function)
        weight = self.gen_weight()
        bias = 0.5
        target = target
        class1_x = sample1[:,0]
        class1_y = sample1[:,1]
        class2_x = sample2[:,0]
        class2_y = sample2[:,1]

        flag_error = 1
        while (flag_error):
            flag_error = 0
            sample = np.concatenate( (sample1,sample2 ), axis=0)
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

#data will take in
if __name__ == '__main__':
    sample1 = np.array([
        [0, 2],
        [3, 4],
        [2, 6],
        [1, -2],
        [4, -3]
    ])

    sample2 = np.array([
        [8, 6],
        [5, 2],
        [9, 3],
        [6, 9],
        [7, -1]
    ])
    target = np.array( [0,0,0,0,0,1,1,1,1,1] )
    
perceptron = perceptron()
perceptron.prediction(sample1, sample2 , target)


